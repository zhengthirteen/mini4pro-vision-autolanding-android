package dji.sampleV5.aircraft.graduation

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import dji.sampleV5.aircraft.R
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.flightcontroller.FlightControlAuthorityChangeReason
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.Stick
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickState
import dji.v5.manager.aircraft.virtualstick.VirtualStickStateListener
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class RealTimeTrackingActivity : AppCompatActivity() {

    private enum class TrackingState {
        IDLE,
        SEARCHING,
        ALIGNING,
        DESCENDING,
        AUTO_LANDING,
        LOST,
        EMERGENCY_STOP
    }

    private enum class SearchMode {
        GLOBAL,
        EAGLE_EYE,
        ROI
    }

    private enum class RunMode {
        PREVIEW,
        CONTROL
    }

    private lateinit var surfaceView: SurfaceView
    private lateinit var overlayView: DetectionOverlayView

    private lateinit var tvHeight: TextView
    private lateinit var tvErrorX: TextView
    private lateinit var tvErrorY: TextView
    private lateinit var tvCmd: TextView
    private lateinit var tvVsState: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnEnable: Button
    private lateinit var btnStop: Button
    private lateinit var btnTakeoff: Button
    private lateinit var btnLand: Button
    private lateinit var btnGimbal: Button

    private var previewSurface: Surface? = null
    private var previewSurfaceWidth = 0
    private var previewSurfaceHeight = 0

    private var runMode = RunMode.PREVIEW
    private var isControlEnabled = false
    private var isGimbalDown = false
    private var autoLandingActive = false
    private var trackingState = TrackingState.IDLE
    private var lastStatusDetail = "待机"

    @Volatile
    private var isActivityAlive = true

    private val pidX = PIDController(maxSpeed = 0.5)
    private val pidY = PIDController(maxSpeed = 0.5)

    private lateinit var yoloDetector: YoloV8Detector

    @Volatile
    private var isYoloReady = false

    private var yuvMat: Mat? = null
    private var rgbMat: Mat? = null
    private var frameProc: Mat? = null
    private var globalDetectMat: Mat? = null

    private val frameLock = Any()
    private var sampledFrameBuffer: ByteArray? = null
    private var sampledFrameWidth = 0
    private var sampledFrameHeight = 0

    private var detectExecutor: ExecutorService? = null
    private val detectInFlight = AtomicBoolean(false)
    private var lastDetectRequestMs = 0L

    private var lastTargetRect: Rect? = null
    private var lastMainTarget: OverlayTarget? = null
    private var trackingLostCount = 0
    private var consecutiveFoundFrames = 0
    private var consecutiveAlignedFrames = 0
    private var consecutiveLostFrames = 0
    private var consecutiveFinalFrames = 0

    private var currentDisplayTargets: List<OverlayTarget> = emptyList()
    private var currentMainTarget: OverlayTarget? = null
    private var currentSearchMode = SearchMode.GLOBAL
    private var currentRoiRect: Rect? = null

    private var lastCmdRoll = 0.0
    private var lastCmdPitch = 0.0
    private var lastCmdYaw = 0.0
    private var lastCmdVert = 0.0
    private var lastBoxRatio = 0.0
    private var lastErrX = 0.0
    private var lastErrY = 0.0

    private var currentUltrasonicHeightM = 0.0
    private var isAircraftFlying = false
    private var landingConfirmationNeeded = false

    private var lastVsEnabled = false
    private var lastVsAuthority = "--"
    private var lastVsReason = "--"

    private var lastDashboardPushMs = 0L

    private var previewFrameCount = 0
    private var detectFrameCount = 0
    private var previewFps = 0.0
    private var detectFps = 0.0
    private var fpsWindowStartMs = SystemClock.uptimeMillis()

    companion object {
        private const val TAG = "RealTimeTracking"

        private const val PROC_W = 960
        private const val PROC_H = 540
        private const val CROP_SIZE = 480
        private const val DETECT_INTERVAL_MS = 300L

        private const val LOST_FRAME_THRESHOLD = 10
        private const val FOUND_FRAME_THRESHOLD = 3
        private const val ALIGN_FRAME_THRESHOLD = 5
        private const val FINAL_FRAME_THRESHOLD = 4

        private const val ALIGN_THRESHOLD_X = 0.08
        private const val ALIGN_THRESHOLD_Y = 0.08
        private const val FINAL_ALIGN_THRESHOLD_X = 0.05
        private const val FINAL_ALIGN_THRESHOLD_Y = 0.05
        private const val FINAL_BOX_RATIO_THRESHOLD = 0.35
        private const val FINAL_HEIGHT_THRESHOLD_M = 1.20

        private const val DASHBOARD_PUSH_INTERVAL_MS = 120L
    }

    private val vsStateListener = object : VirtualStickStateListener {
        override fun onVirtualStickStateUpdate(stickState: VirtualStickState) {
            lastVsEnabled = stickState.isVirtualStickEnable()
            lastVsAuthority = stickState.getCurrentFlightControlAuthorityOwner().name
            updateVsStateText()
        }

        override fun onChangeReasonUpdate(reason: FlightControlAuthorityChangeReason) {
            lastVsReason = reason.name
            updateVsStateText()

            if (isControlEnabled && reason != FlightControlAuthorityChangeReason.MSDK_REQUEST) {
                stopTracking("控制权变化: ${reason.name}", TrackingState.EMERGENCY_STOP)
            }
        }
    }

    private val surfaceCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) {
            previewSurface = holder.surface
        }

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            previewSurfaceWidth = width
            previewSurfaceHeight = height
            attachPreviewSurface()
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            detachPreviewSurface()
            previewSurface = null
            previewSurfaceWidth = 0
            previewSurfaceHeight = 0
        }
    }

    private val frameListener = object : ICameraStreamManager.CameraFrameListener {
        override fun onFrame(
            data: ByteArray,
            offset: Int,
            length: Int,
            width: Int,
            height: Int,
            format: ICameraStreamManager.FrameFormat
        ) {
            if (!isActivityAlive) return
            if (data.isEmpty() || width <= 0 || height <= 0) return

            tickPreviewFrame()

            if (!isYoloReady) return

            val now = SystemClock.uptimeMillis()
            if (now - lastDetectRequestMs < DETECT_INTERVAL_MS) return
            if (!detectInFlight.compareAndSet(false, true)) return

            lastDetectRequestMs = now

            synchronized(frameLock) {
                if (sampledFrameBuffer == null || sampledFrameBuffer!!.size != length) {
                    sampledFrameBuffer = ByteArray(length)
                }
                System.arraycopy(data, offset, sampledFrameBuffer!!, 0, length)
                sampledFrameWidth = width
                sampledFrameHeight = height
            }

            detectExecutor?.execute {
                try {
                    detectLatestSample()
                } catch (e: Exception) {
                    Log.e(TAG, "detectLatestSample error: ${e.message}", e)
                } finally {
                    detectInFlight.set(false)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_tracking)

        detectExecutor = Executors.newSingleThreadExecutor()

        initView()
        setupListeners()
        initVirtualStickListener()
        initTelemetryListeners()

        if (OpenCVLoader.initDebug()) {
            initYoloModel()
            initDJIStream()
            showStatus("OpenCV 初始化成功")
        } else {
            showStatus("OpenCV 加载失败")
        }
    }

    private fun initView() {
        surfaceView = findViewById(R.id.main_surface)
        overlayView = findViewById(R.id.overlay_view)

        tvHeight = findViewById(R.id.tv_height)
        tvErrorX = findViewById(R.id.tv_error_x)
        tvErrorY = findViewById(R.id.tv_error_y)
        tvCmd = findViewById(R.id.tv_cmd)
        tvVsState = findViewById(R.id.tv_vs_state)
        tvStatus = findViewById(R.id.tv_status)

        btnEnable = findViewById(R.id.btn_enable_control)
        btnStop = findViewById(R.id.btn_emergency_stop)
        btnTakeoff = findViewById(R.id.btn_takeoff)
        btnLand = findViewById(R.id.btn_land)
        btnGimbal = findViewById(R.id.btn_gimbal_down)

        surfaceView.holder.addCallback(surfaceCallback)

        updateDashboard(
            ratio = 0.0,
            errX = 0.0,
            errY = 0.0,
            cmdRoll = 0.0,
            cmdPitch = 0.0,
            cmdVert = 0.0,
            detail = "待机",
            force = true
        )
        updateVsStateText()
        overlayView.clearOverlay()
        updateControlUi()
    }

    private fun setupListeners() {
        btnTakeoff.setOnClickListener {
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStartTakeoff),
                null
            )
            showStatus("已发送自动起飞")
        }

        btnLand.setOnClickListener {
            if (autoLandingActive) {
                showToast("自动降落已经接管")
                return@setOnClickListener
            }
            startAutoLandingTransfer("手动移交自动降落")
        }

        btnGimbal.setOnClickListener {
            if (isGimbalDown) {
                controlGimbal(0.0)
                isGimbalDown = false
                btnGimbal.text = "云台朝下"
                showStatus("云台回正")
            } else {
                controlGimbal(-90.0)
                isGimbalDown = true
                btnGimbal.text = "云台回正"
                showStatus("云台朝下")
            }
        }

        btnEnable.setOnClickListener {
            if (!isYoloReady) {
                showToast("AI 模型还没准备好")
                return@setOnClickListener
            }

            if (autoLandingActive) {
                showToast("自动降落接管中，不能重新进入控制")
                return@setOnClickListener
            }

            if (isControlEnabled) {
                showToast("控制模式已经在运行")
                return@setOnClickListener
            }

            prepareForControlEntry()

            if (!isGimbalDown) {
                showToast("正在调整云台...")
                controlGimbal(-90.0)
                isGimbalDown = true
                btnGimbal.text = "云台回正"
                surfaceView.postDelayed({ startVirtualStick() }, 1200)
            } else {
                startVirtualStick()
            }
        }

        btnStop.setOnClickListener {
            if (autoLandingActive) {
                stopAutoLandingAndEnterPreview("已停止自动降落")
            } else {
                stopTracking("紧急停止", TrackingState.EMERGENCY_STOP)
            }
        }
    }

    private fun initVirtualStickListener() {
        try {
            VirtualStickManager.getInstance().setVirtualStickStateListener(vsStateListener)
        } catch (e: Exception) {
            Log.w(TAG, "setVirtualStickStateListener failed: ${e.message}")
        }
    }

    private fun initTelemetryListeners() {
        val keyManager = KeyManager.getInstance()
        keyManager.listen(KeyTools.createKey(FlightControllerKey.KeyUltrasonicHeight), this, true) { _, newValue: Int? ->
            currentUltrasonicHeightM = ((newValue ?: 0).toDouble() / 10.0).coerceAtLeast(0.0)
        }
        keyManager.listen(KeyTools.createKey(FlightControllerKey.KeyIsLandingConfirmationNeeded), this, true) { _, newValue: Boolean? ->
            landingConfirmationNeeded = newValue == true
            if (autoLandingActive && landingConfirmationNeeded) {
                confirmLandingIfNeeded()
            }
        }
        keyManager.listen(KeyTools.createKey(FlightControllerKey.KeyIsFlying), this, true) { _, newValue: Boolean? ->
            isAircraftFlying = newValue == true
            if (autoLandingActive && !isAircraftFlying) {
                handleAutoLandingCompleted()
            }
        }
    }

    private fun initYoloModel() {
        yoloDetector = YoloV8Detector()
        showStatus("正在加载 AI 模型...")
        Thread {
            try {
                val modelPath = AssetUtils.getAssetFilePath(this, "yolov8n.onnx")
                isYoloReady = yoloDetector.init(modelPath)
                runOnUiThread {
                    if (!isActivityAlive) return@runOnUiThread
                    if (isYoloReady) {
                        showStatus("AI 模型加载成功")
                        showToast("YOLO Ready")
                    } else {
                        showStatus("模型加载失败")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "initYoloModel error: ${e.message}", e)
                runOnUiThread {
                    if (!isActivityAlive) return@runOnUiThread
                    showStatus("模型加载异常: ${e.message}")
                }
            }
        }.start()
    }

    private fun initDJIStream() {
        val streamManager = MediaDataCenter.getInstance().cameraStreamManager
        try {
            streamManager.setKeepAliveDecoding(true)
        } catch (e: Exception) {
            Log.w(TAG, "setKeepAliveDecoding(true) failed: ${e.message}")
        }

        streamManager.addFrameListener(
            ComponentIndexType.LEFT_OR_MAIN,
            ICameraStreamManager.FrameFormat.NV21,
            frameListener
        )
    }

    private fun attachPreviewSurface() {
        val surface = previewSurface ?: return
        if (previewSurfaceWidth <= 0 || previewSurfaceHeight <= 0) return

        try {
            MediaDataCenter.getInstance().cameraStreamManager.putCameraStreamSurface(
                ComponentIndexType.LEFT_OR_MAIN,
                surface,
                previewSurfaceWidth,
                previewSurfaceHeight,
                ICameraStreamManager.ScaleType.CENTER_INSIDE
            )
        } catch (e: Exception) {
            Log.e(TAG, "attachPreviewSurface error: ${e.message}", e)
        }
    }

    private fun detachPreviewSurface() {
        val surface = previewSurface ?: return
        try {
            MediaDataCenter.getInstance().cameraStreamManager.removeCameraStreamSurface(surface)
        } catch (e: Exception) {
            Log.w(TAG, "removeCameraStreamSurface failed: ${e.message}")
        }
    }

    private fun prepareForControlEntry() {
        pidX.reset()
        pidY.reset()
        autoLandingActive = false
        landingConfirmationNeeded = false
        runMode = RunMode.PREVIEW

        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0

        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0
        lastStatusDetail = "准备进入控制模式"

        updateDashboard(lastBoxRatio, lastErrX, lastErrY, 0.0, 0.0, 0.0, lastStatusDetail, force = true)
        updateControlUi()
    }

    private fun startVirtualStick() {
        try {
            VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false)
        } catch (e: Exception) {
            Log.w(TAG, "setVirtualStickAdvancedModeEnabled(false) failed: ${e.message}")
        }

        VirtualStickManager.getInstance().enableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                isControlEnabled = true
                autoLandingActive = false
                runMode = RunMode.CONTROL
                pidX.reset()
                pidY.reset()
                setTrackingState(TrackingState.SEARCHING, "已获取控制权，开始搜索目标")
                updateControlUi()
            }

            override fun onFailure(error: IDJIError) {
                isControlEnabled = false
                autoLandingActive = false
                runMode = RunMode.PREVIEW
                setTrackingState(TrackingState.IDLE, "控制权获取失败: ${error.description()}")
                updateControlUi()
            }
        })
    }

    private fun stopTracking(reason: String, nextState: TrackingState = TrackingState.EMERGENCY_STOP) {
        sendVelocityCommand(0.0, 0.0, 0.0, 0.0)

        isControlEnabled = false
        autoLandingActive = false
        runMode = RunMode.PREVIEW
        pidX.reset()
        pidY.reset()

        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0

        try {
            VirtualStickManager.getInstance().disableVirtualStick(null)
        } catch (e: Exception) {
            Log.w(TAG, "disableVirtualStick failed: ${e.message}")
        }

        setTrackingState(nextState, reason)
        updateControlUi()
    }

    private fun stopAutoLandingAndEnterPreview(reason: String) {
        try {
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStopAutoLanding),
                null
            )
        } catch (e: Exception) {
            Log.w(TAG, "stopAutoLanding failed: ${e.message}")
        }

        autoLandingActive = false
        runMode = RunMode.PREVIEW
        setTrackingState(TrackingState.IDLE, reason)
        updateControlUi()
    }

    private fun startAutoLandingTransfer(detail: String) {
        sendVelocityCommand(0.0, 0.0, 0.0, 0.0)

        if (isControlEnabled) {
            try {
                VirtualStickManager.getInstance().disableVirtualStick(null)
            } catch (e: Exception) {
                Log.w(TAG, "disableVirtualStick before autoland failed: ${e.message}")
            }
        }

        isControlEnabled = false
        autoLandingActive = true
        runMode = RunMode.PREVIEW
        pidX.reset()
        pidY.reset()
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0

        setTrackingState(TrackingState.AUTO_LANDING, detail)
        updateControlUi()

        try {
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStartAutoLanding),
                null
            )
        } catch (e: Exception) {
            Log.e(TAG, "startAutoLandingTransfer error: ${e.message}", e)
            autoLandingActive = false
            setTrackingState(TrackingState.IDLE, "自动降落下发失败: ${e.message}")
            updateControlUi()
        }
    }

    private fun confirmLandingIfNeeded() {
        try {
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyConfirmLanding),
                null
            )
            landingConfirmationNeeded = false
            showStatus("低空确认已发送，继续自动降落")
        } catch (e: Exception) {
            Log.w(TAG, "confirmLandingIfNeeded failed: ${e.message}")
        }
    }

    private fun handleAutoLandingCompleted() {
        autoLandingActive = false
        runMode = RunMode.PREVIEW
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0
        setTrackingState(TrackingState.IDLE, "自动降落完成")
        updateControlUi()
        updateDashboard(lastBoxRatio, lastErrX, lastErrY, 0.0, 0.0, 0.0, lastStatusDetail, force = true)
    }

    private fun resetTrackingSession() {
        pidX.reset()
        pidY.reset()

        runMode = RunMode.PREVIEW
        isControlEnabled = false
        autoLandingActive = false
        trackingState = TrackingState.IDLE
        lastStatusDetail = "已重置会话"

        lastTargetRect = null
        lastMainTarget = null
        trackingLostCount = 0
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0

        currentDisplayTargets = emptyList()
        currentMainTarget = null
        currentSearchMode = SearchMode.GLOBAL
        currentRoiRect = null

        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0
        lastBoxRatio = 0.0
        lastErrX = 0.0
        lastErrY = 0.0

        previewFrameCount = 0
        detectFrameCount = 0
        previewFps = 0.0
        detectFps = 0.0
        fpsWindowStartMs = SystemClock.uptimeMillis()

        updateDashboard(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lastStatusDetail, force = true)
        updateControlUi()

        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            overlayView.clearOverlay()
        }
    }

    private fun controlGimbal(pitch: Double) {
        val key = KeyTools.createKey(GimbalKey.KeyRotateByAngle)
        val rotation = GimbalAngleRotation().apply {
            mode = GimbalAngleRotationMode.ABSOLUTE_ANGLE
            this.pitch = pitch
        }
        KeyManager.getInstance().performAction(key, rotation, null)
    }

    private fun detectLatestSample() {
        val localFrame: ByteArray
        val inputWidth: Int
        val inputHeight: Int

        synchronized(frameLock) {
            val buffer = sampledFrameBuffer ?: return
            localFrame = buffer.copyOf()
            inputWidth = sampledFrameWidth
            inputHeight = sampledFrameHeight
        }

        ensureDetectionBuffers(inputWidth, inputHeight)

        val yuvHeight = inputHeight + inputHeight / 2
        if (yuvMat == null || yuvMat!!.width() != inputWidth || yuvMat!!.height() != yuvHeight) {
            yuvMat?.release()
            yuvMat = Mat(yuvHeight, inputWidth, CvType.CV_8UC1)
        }

        if (rgbMat == null || rgbMat!!.width() != inputWidth || rgbMat!!.height() != inputHeight) {
            rgbMat?.release()
            rgbMat = Mat(inputHeight, inputWidth, CvType.CV_8UC3)
        }

        yuvMat!!.put(0, 0, localFrame)
        Imgproc.cvtColor(yuvMat!!, rgbMat!!, Imgproc.COLOR_YUV2RGB_NV21)
        Imgproc.resize(rgbMat!!, frameProc!!, Size(PROC_W.toDouble(), PROC_H.toDouble()))

        runDetectionPipeline(PROC_W, PROC_H)
        tickDetectFrame()
        pushOverlayToUi()
    }

    private fun ensureDetectionBuffers(inputWidth: Int, inputHeight: Int) {
        if (frameProc == null || frameProc!!.width() != PROC_W || frameProc!!.height() != PROC_H) {
            frameProc?.release()
            frameProc = Mat(PROC_H, PROC_W, CvType.CV_8UC3)
        }

        if (globalDetectMat == null || globalDetectMat!!.width() != CROP_SIZE || globalDetectMat!!.height() != CROP_SIZE) {
            globalDetectMat?.release()
            globalDetectMat = Mat(CROP_SIZE, CROP_SIZE, CvType.CV_8UC3)
        }

        if (rgbMat == null || rgbMat!!.width() != inputWidth || rgbMat!!.height() != inputHeight) {
            rgbMat?.release()
            rgbMat = Mat(inputHeight, inputWidth, CvType.CV_8UC3)
        }
    }

    private fun runDetectionPipeline(imgW: Int, imgH: Int) {
        val frame = frameProc ?: return

        var results: List<DetectionResult> = emptyList()
        var roiForThisFrame: Rect? = null
        var searchModeForThisFrame = SearchMode.GLOBAL

        if (lastTargetRect != null && trackingLostCount < LOST_FRAME_THRESHOLD) {
            val roiRect = buildCropRect(lastTargetRect!!, imgW, imgH)
            val roiMat = Mat(frame, roiRect)
            try {
                results = yoloDetector.detect(roiMat)
            } finally {
                roiMat.release()
            }

            if (results.isNotEmpty()) {
                roiForThisFrame = roiRect
                searchModeForThisFrame = SearchMode.ROI
                trackingLostCount = 0
            } else {
                trackingLostCount++
            }
        }

        if (results.isEmpty()) {
            val centerRect = Rect(
                (imgW - CROP_SIZE) / 2,
                (imgH - CROP_SIZE) / 2,
                CROP_SIZE,
                CROP_SIZE
            )
            val centerMat = Mat(frame, centerRect)
            try {
                results = yoloDetector.detect(centerMat)
            } finally {
                centerMat.release()
            }

            if (results.isNotEmpty()) {
                roiForThisFrame = centerRect
                searchModeForThisFrame = SearchMode.EAGLE_EYE
                trackingLostCount = 0
            }
        }

        if (results.isEmpty()) {
            Imgproc.resize(frame, globalDetectMat!!, Size(CROP_SIZE.toDouble(), CROP_SIZE.toDouble()))
            results = yoloDetector.detect(globalDetectMat!!)
            roiForThisFrame = null
            searchModeForThisFrame = SearchMode.GLOBAL

            if (results.isNotEmpty()) {
                trackingLostCount = 0
            }
        }

        currentRoiRect = roiForThisFrame
        currentSearchMode = searchModeForThisFrame

        val allTargets = results.map { result ->
            val displayRect = restoreRectToProc(result.rect, roiForThisFrame, imgW, imgH)
            OverlayTarget(
                label = result.label,
                confidence = result.confidence,
                rect = displayRect
            )
        }

        val selected = MainTargetSelector.selectMainTarget(allTargets, lastMainTarget, imgW, imgH)
        currentMainTarget = selected?.deepCopy(isMainTarget = true)
        currentDisplayTargets = allTargets.map { target ->
            target.deepCopy(isMainTarget = selected != null && isSameRect(target.rect, selected.rect) && target.label == selected.label)
        }

        if (selected != null) {
            lastMainTarget = selected.deepCopy()
            lastTargetRect = Rect(selected.rect.x, selected.rect.y, selected.rect.width, selected.rect.height)

            if (runMode == RunMode.CONTROL) {
                handleTargetFound(selected, imgW, imgH)
            } else {
                holdPreviewObservation(selected, imgW, imgH)
            }
        } else {
            if (runMode == RunMode.CONTROL) {
                handleTargetLost()
            } else {
                holdPreviewNoTarget()
            }
        }
    }

    private fun holdPreviewObservation(target: OverlayTarget, imgW: Int, imgH: Int) {
        lastBoxRatio = target.rect.width.toDouble() / imgW.toDouble()
        lastErrX = (target.centerX - imgW / 2.0) / (imgW / 2.0)
        lastErrY = (target.centerY - imgH / 2.0) / (imgH / 2.0)
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0

        if (!shouldKeepCurrentStateInPreview()) {
            trackingState = TrackingState.IDLE
            lastStatusDetail = "预览模式：仅检测显示"
        }

        updateDashboard(lastBoxRatio, lastErrX, lastErrY, 0.0, 0.0, 0.0, lastStatusDetail)
    }

    private fun holdPreviewNoTarget() {
        lastErrX = 0.0
        lastErrY = 0.0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0

        if (!shouldKeepCurrentStateInPreview()) {
            trackingState = TrackingState.IDLE
            lastStatusDetail = "预览模式：未发现目标"
        }

        updateDashboard(lastBoxRatio, 0.0, 0.0, 0.0, 0.0, 0.0, lastStatusDetail)
    }

    private fun pushOverlayToUi() {
        val targetsCopy = currentDisplayTargets.map { it.deepCopy() }
        val roiCopy = currentRoiRect?.let { Rect(it.x, it.y, it.width, it.height) }
        val modeText = buildOverlayModeText()

        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            overlayView.updateOverlay(
                procWidth = PROC_W,
                procHeight = PROC_H,
                targets = targetsCopy,
                roiRect = roiCopy,
                modeText = modeText
            )
        }
    }

    private fun handleTargetFound(target: OverlayTarget, imgW: Int, imgH: Int) {
        val rect = target.rect
        val normErrorX = (target.centerX - imgW / 2.0) / (imgW / 2.0)
        val normErrorY = (target.centerY - imgH / 2.0) / (imgH / 2.0)
        val boxRatio = rect.width.toDouble() / imgW.toDouble()

        lastBoxRatio = boxRatio
        lastErrX = normErrorX
        lastErrY = normErrorY

        consecutiveLostFrames = 0
        consecutiveFoundFrames++

        if (trackingState == TrackingState.LOST || trackingState == TrackingState.IDLE) {
            setTrackingState(TrackingState.SEARCHING, "重新发现目标，等待稳定")
        }

        val alignedNow = abs(normErrorX) < ALIGN_THRESHOLD_X && abs(normErrorY) < ALIGN_THRESHOLD_Y
        val finalAlignedNow = abs(normErrorX) < FINAL_ALIGN_THRESHOLD_X && abs(normErrorY) < FINAL_ALIGN_THRESHOLD_Y

        if (trackingState == TrackingState.SEARCHING && consecutiveFoundFrames >= FOUND_FRAME_THRESHOLD) {
            setTrackingState(TrackingState.ALIGNING, "目标稳定，开始对准")
        }

        if (trackingState == TrackingState.ALIGNING) {
            if (alignedNow) {
                consecutiveAlignedFrames++
                if (consecutiveAlignedFrames >= ALIGN_FRAME_THRESHOLD) {
                    consecutiveFinalFrames = 0
                    setTrackingState(TrackingState.DESCENDING, "对准稳定，开始下降")
                }
            } else {
                consecutiveAlignedFrames = 0
            }
        } else if (trackingState == TrackingState.DESCENDING) {
            if (!alignedNow) {
                consecutiveAlignedFrames = 0
                consecutiveFinalFrames = 0
                setTrackingState(TrackingState.ALIGNING, "偏差变大，回到对准")
            } else if (finalAlignedNow && isReadyForAutoLanding(boxRatio)) {
                consecutiveFinalFrames++
                if (consecutiveFinalFrames >= FINAL_FRAME_THRESHOLD) {
                    startAutoLandingTransfer("进入最终阶段，移交自动降落")
                    return
                }
            } else {
                consecutiveFinalFrames = 0
            }
        }

        val vRoll = pidX.calculate(normErrorX * 2.0)
        val vPitch = pidY.calculate(-normErrorY * 2.0)
        val vVert = if (trackingState == TrackingState.DESCENDING) computeVerticalSpeed(boxRatio) else 0.0

        when (trackingState) {
            TrackingState.SEARCHING -> {
                sendVelocityCommand(0.0, 0.0, 0.0, 0.0)
                updateDashboard(boxRatio, normErrorX, normErrorY, 0.0, 0.0, 0.0, "已发现目标，等待稳定")
            }

            TrackingState.ALIGNING -> {
                sendVelocityCommand(vRoll, vPitch, 0.0, 0.0)
                updateDashboard(boxRatio, normErrorX, normErrorY, vRoll, vPitch, 0.0, "正在对准")
            }

            TrackingState.DESCENDING -> {
                sendVelocityCommand(vRoll, vPitch, 0.0, vVert)
                updateDashboard(boxRatio, normErrorX, normErrorY, vRoll, vPitch, vVert, "视觉下降中")
            }

            TrackingState.AUTO_LANDING,
            TrackingState.LOST,
            TrackingState.EMERGENCY_STOP,
            TrackingState.IDLE -> {
                sendVelocityCommand(0.0, 0.0, 0.0, 0.0)
                updateDashboard(boxRatio, normErrorX, normErrorY, 0.0, 0.0, 0.0, "未处于自动控制")
            }
        }
    }

    private fun handleTargetLost() {
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveFinalFrames = 0
        consecutiveLostFrames++

        if (consecutiveLostFrames > LOST_FRAME_THRESHOLD) {
            trackingLostCount = LOST_FRAME_THRESHOLD
            lastTargetRect = null
            lastMainTarget = null
            currentMainTarget = null
            currentDisplayTargets = emptyList()
            currentRoiRect = null
            currentSearchMode = SearchMode.GLOBAL

            if (isControlEnabled) {
                setTrackingState(TrackingState.LOST, "目标丢失")
            }
        } else if (isControlEnabled && trackingState != TrackingState.SEARCHING) {
            setTrackingState(TrackingState.SEARCHING, "短时丢失，继续搜索")
        }

        if (isControlEnabled) {
            sendVelocityCommand(0.0, 0.0, 0.0, 0.0)
        }

        updateDashboard(lastBoxRatio, 0.0, 0.0, 0.0, 0.0, 0.0, lastStatusDetail)
    }

    private fun buildCropRect(lastRect: Rect, imgW: Int, imgH: Int): Rect {
        val cx = lastRect.x + lastRect.width / 2
        val cy = lastRect.y + lastRect.height / 2

        val cropX = (cx - CROP_SIZE / 2).coerceIn(0, max(0, imgW - CROP_SIZE))
        val cropY = (cy - CROP_SIZE / 2).coerceIn(0, max(0, imgH - CROP_SIZE))

        return Rect(cropX, cropY, CROP_SIZE, CROP_SIZE)
    }

    private fun restoreRectToProc(rawRect: Rect, roi: Rect?, imgW: Int, imgH: Int): Rect {
        return if (roi != null) {
            Rect(
                rawRect.x + roi.x,
                rawRect.y + roi.y,
                rawRect.width,
                rawRect.height
            )
        } else {
            val scaleX = imgW.toDouble() / CROP_SIZE.toDouble()
            val scaleY = imgH.toDouble() / CROP_SIZE.toDouble()
            Rect(
                (rawRect.x * scaleX).toInt(),
                (rawRect.y * scaleY).toInt(),
                (rawRect.width * scaleX).toInt(),
                (rawRect.height * scaleY).toInt()
            )
        }
    }

    private fun computeVerticalSpeed(boxRatio: Double): Double {
        return when {
            boxRatio < 0.10 -> -0.50
            boxRatio < 0.25 -> -0.25
            boxRatio < 0.40 -> -0.15
            else -> -0.08
        }
    }

    private fun sendVelocityCommand(
        rollSpeed: Double,
        pitchSpeed: Double,
        yawSpeed: Double,
        verticalSpeed: Double
    ) {
        lastCmdRoll = rollSpeed
        lastCmdPitch = pitchSpeed
        lastCmdYaw = yawSpeed
        lastCmdVert = verticalSpeed

        if (!isControlEnabled) return

        val speedToStickGain = 660.0
        val rollStickVal = (rollSpeed * speedToStickGain).toInt()
        val pitchStickVal = (pitchSpeed * speedToStickGain).toInt()
        val yawStickVal = (yawSpeed * speedToStickGain).toInt()
        val verticalStickVal = (verticalSpeed * speedToStickGain).toInt()

        val maxStick = Stick.MAX_STICK_POSITION_ABS
        val r = clamp(rollStickVal, -maxStick, maxStick)
        val p = clamp(pitchStickVal, -maxStick, maxStick)
        val y = clamp(yawStickVal, -maxStick, maxStick)
        val v = clamp(verticalStickVal, -maxStick, maxStick)

        try {
            val vsManager = VirtualStickManager.getInstance()
            vsManager.rightStick?.apply {
                horizontalPosition = r
                verticalPosition = p
            }
            vsManager.leftStick?.apply {
                horizontalPosition = y
                verticalPosition = v
            }
        } catch (e: Exception) {
            Log.e(TAG, "sendVelocityCommand error: ${e.message}", e)
        }
    }

    private fun clamp(value: Int, minValue: Int, maxValue: Int): Int {
        return max(minValue, min(maxValue, value))
    }

    private fun setTrackingState(newState: TrackingState, detail: String) {
        trackingState = newState
        lastStatusDetail = detail
        showStatus(detail)
    }

    private fun stateLabel(state: TrackingState): String {
        return when (state) {
            TrackingState.IDLE -> "IDLE"
            TrackingState.SEARCHING -> "SEARCHING"
            TrackingState.ALIGNING -> "ALIGNING"
            TrackingState.DESCENDING -> "DESCENDING"
            TrackingState.AUTO_LANDING -> "AUTO_LANDING"
            TrackingState.LOST -> "LOST"
            TrackingState.EMERGENCY_STOP -> "STOP"
        }
    }

    private fun buildOverlayModeText(): String {
        val searchText = when (currentSearchMode) {
            SearchMode.ROI -> "ROI TRACKING"
            SearchMode.EAGLE_EYE -> "EAGLE EYE"
            SearchMode.GLOBAL -> "GLOBAL SEARCH"
        }
        val runText = when (runMode) {
            RunMode.PREVIEW -> "PREVIEW"
            RunMode.CONTROL -> "CONTROL"
        }
        return "$searchText | $runText"
    }

    private fun buildStatusLine(detail: String): String {
        return "状态: ${stateLabel(trackingState)} | 模式: ${runMode.name} | $detail | FPS: %.1f | DET: %.1fHz"
            .format(previewFps, detectFps)
    }

    @Synchronized
    private fun tickPreviewFrame() {
        previewFrameCount++
        updateFpsWindowLocked()
    }

    @Synchronized
    private fun tickDetectFrame() {
        detectFrameCount++
        updateFpsWindowLocked()
    }

    private fun updateFpsWindowLocked() {
        val now = SystemClock.uptimeMillis()
        val dt = now - fpsWindowStartMs
        if (dt >= 1000L) {
            previewFps = previewFrameCount * 1000.0 / dt
            detectFps = detectFrameCount * 1000.0 / dt
            previewFrameCount = 0
            detectFrameCount = 0
            fpsWindowStartMs = now
        }
    }

    private fun updateDashboard(
        ratio: Double,
        errX: Double,
        errY: Double,
        cmdRoll: Double,
        cmdPitch: Double,
        cmdVert: Double,
        detail: String,
        force: Boolean = false
    ) {
        lastStatusDetail = detail
        val now = SystemClock.uptimeMillis()
        if (!force && now - lastDashboardPushMs < DASHBOARD_PUSH_INTERVAL_MS) return
        lastDashboardPushMs = now

        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            tvHeight.text = "高度: %.2fm | 目标占比: %.2f".format(currentUltrasonicHeightM, ratio)
            tvErrorX.text = "ErrX: %.3f".format(errX)
            tvErrorY.text = "ErrY: %.3f".format(errY)
            tvCmd.text = "Cmd R/P/V: %.2f / %.2f / %.2f".format(cmdRoll, cmdPitch, cmdVert)
            tvStatus.text = buildStatusLine(detail)
        }
    }

    private fun updateVsStateText() {
        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            tvVsState.text = "VS: ${if (lastVsEnabled) "ON" else "OFF"} | Authority: $lastVsAuthority | Reason: $lastVsReason"
        }
    }

    private fun updateControlUi() {
        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            when {
                autoLandingActive -> {
                    btnEnable.text = "自动降落接管中"
                    btnEnable.isEnabled = false
                }
                isControlEnabled -> {
                    btnEnable.text = "控制中..."
                    btnEnable.isEnabled = false
                }
                else -> {
                    btnEnable.text = "进入控制模式"
                    btnEnable.isEnabled = true
                }
            }
        }
    }

    private fun shouldKeepCurrentStateInPreview(): Boolean {
        return autoLandingActive || trackingState == TrackingState.AUTO_LANDING || trackingState == TrackingState.EMERGENCY_STOP
    }

    private fun isReadyForAutoLanding(boxRatio: Double): Boolean {
        val heightReady = currentUltrasonicHeightM > 0.05 && currentUltrasonicHeightM <= FINAL_HEIGHT_THRESHOLD_M
        val ratioReady = boxRatio >= FINAL_BOX_RATIO_THRESHOLD
        return ratioReady || heightReady
    }

    private fun isSameRect(a: Rect, b: Rect): Boolean {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height
    }

    private fun showStatus(detail: String) {
        lastStatusDetail = detail
        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            tvStatus.text = buildStatusLine(detail)
        }
    }

    private fun showToast(msg: String) {
        runOnUiThread {
            if (!isActivityAlive) return@runOnUiThread
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        isActivityAlive = false

        if (autoLandingActive) {
            stopAutoLandingAndEnterPreview("页面关闭，停止自动降落监听")
        } else {
            stopTracking("页面关闭，释放控制", TrackingState.IDLE)
        }

        try {
            MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(frameListener)
        } catch (e: Exception) {
            Log.w(TAG, "removeFrameListener failed: ${e.message}")
        }

        detachPreviewSurface()

        try {
            VirtualStickManager.getInstance().removeVirtualStickStateListener(vsStateListener)
        } catch (e: Exception) {
            Log.w(TAG, "removeVirtualStickStateListener failed: ${e.message}")
        }

        try {
            KeyManager.getInstance().cancelListen(this)
        } catch (e: Exception) {
            Log.w(TAG, "cancelListen failed: ${e.message}")
        }

        detectExecutor?.shutdownNow()
        detectExecutor = null

        yuvMat?.release()
        rgbMat?.release()
        frameProc?.release()
        globalDetectMat?.release()

        super.onDestroy()
    }
}
