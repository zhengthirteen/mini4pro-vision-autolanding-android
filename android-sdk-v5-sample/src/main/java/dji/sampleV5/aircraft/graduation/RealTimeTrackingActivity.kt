package dji.sampleV5.aircraft.graduation

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
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
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
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
        LOST,
        EMERGENCY_STOP
    }

    private enum class SearchMode {
        GLOBAL,
        EAGLE_EYE,
        ROI
    }

    // UI
    private lateinit var mainView: ImageView
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

    // 状态
    private var isControlEnabled = false
    private var isGimbalDown = false
    private var trackingState = TrackingState.IDLE

    // PID
    private val pidX = PIDController(maxSpeed = 0.5)
    private val pidY = PIDController(maxSpeed = 0.5)

    // YOLO
    private lateinit var yoloDetector: YoloV8Detector
    private var isYoloReady = false

    // OpenCV
    private var yuvMat: Mat? = null
    private var rgbMat: Mat? = null
    private var frame720p: Mat? = null
    private var globalDetectMat: Mat? = null
    private val isProcessing = AtomicBoolean(false)

    // 检测 / 追踪
    private var frameCounter = 0
    private var lastTargetRect: Rect? = null
    private var trackingLostCount = 0
    private var consecutiveFoundFrames = 0
    private var consecutiveAlignedFrames = 0
    private var consecutiveLostFrames = 0

    private var currentDisplayRect: Rect? = null
    private var currentDisplayLabel: String? = null
    private var currentDisplayConfidence: Float = 0f
    private var currentSearchMode = SearchMode.GLOBAL
    private var currentRoiRect: Rect? = null

    // 控制显示
    private var lastCmdRoll = 0.0
    private var lastCmdPitch = 0.0
    private var lastCmdYaw = 0.0
    private var lastCmdVert = 0.0
    private var lastBoxRatio = 0.0
    private var lastErrX = 0.0
    private var lastErrY = 0.0

    // VS 状态
    private var lastVsEnabled = false
    private var lastVsAuthority = "--"
    private var lastVsReason = "--"

    private val controllableLabels = setOf(
        "land_h",
        "blue_circle_2",
        "green_circle_3",
        "red_circle_1"
    )

    companion object {
        private const val TAG = "RealTimeTracking"
        private const val PROC_W = 1280
        private const val PROC_H = 720
        private const val CROP_SIZE = 640

        private const val DETECT_INTERVAL = 3
        private const val LOST_FRAME_THRESHOLD = 10
        private const val FOUND_FRAME_THRESHOLD = 3
        private const val ALIGN_FRAME_THRESHOLD = 5

        private const val ALIGN_THRESHOLD_X = 0.08
        private const val ALIGN_THRESHOLD_Y = 0.08
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

            // 控制权被 RC 或系统夺回时，立即停掉 AI
            if (isControlEnabled && reason != FlightControlAuthorityChangeReason.MSDK_REQUEST) {
                stopTracking("控制权变化: ${reason.name}", TrackingState.EMERGENCY_STOP)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_tracking)

        initView()
        setupListeners()
        initVirtualStickListener()

        if (OpenCVLoader.initDebug()) {
            initYoloModel()
            initDJIStream()
            showStatus("状态: IDLE | OpenCV 初始化成功")
        } else {
            showStatus("状态: IDLE | OpenCV 加载失败")
        }
    }

    private fun initView() {
        mainView = findViewById(R.id.main_view)
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

        updateDashboard(
            ratio = 0.0,
            errX = 0.0,
            errY = 0.0,
            cmdRoll = 0.0,
            cmdPitch = 0.0,
            cmdVert = 0.0,
            detail = "待机"
        )
        updateVsStateText()
    }

    private fun initVirtualStickListener() {
        try {
            VirtualStickManager.getInstance().setVirtualStickStateListener(vsStateListener)
        } catch (e: Exception) {
            Log.w(TAG, "setVirtualStickStateListener failed: ${e.message}")
        }
    }

    private fun initYoloModel() {
        yoloDetector = YoloV8Detector()
        showStatus("状态: IDLE | 正在加载 AI 模型...")
        Thread {
            try {
                val modelPath = AssetUtils.getAssetFilePath(this, "yolov8n.onnx")
                isYoloReady = yoloDetector.init(modelPath)
                runOnUiThread {
                    if (isYoloReady) {
                        showStatus("状态: IDLE | AI 模型加载成功")
                        showToast("YOLO Ready")
                    } else {
                        showStatus("状态: IDLE | 模型加载失败")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "initYoloModel error: ${e.message}", e)
                runOnUiThread {
                    showStatus("状态: IDLE | 模型加载异常: ${e.message}")
                }
            }
        }.start()
    }

    private fun setupListeners() {
        btnTakeoff.setOnClickListener {
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStartTakeoff),
                null
            )
            showStatus("状态: ${stateLabel(trackingState)} | 已发送自动起飞")
        }

        btnLand.setOnClickListener {
            stopTracking("移交自动降落", TrackingState.IDLE)
            KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStartAutoLanding),
                null
            )
            showStatus("状态: IDLE | 已发送自动降落")
        }

        btnGimbal.setOnClickListener {
            if (isGimbalDown) {
                controlGimbal(0.0)
                isGimbalDown = false
                btnGimbal.text = "云台朝下"
                showStatus("状态: ${stateLabel(trackingState)} | 云台回正")
            } else {
                controlGimbal(-90.0)
                isGimbalDown = true
                btnGimbal.text = "云台回正"
                showStatus("状态: ${stateLabel(trackingState)} | 云台朝下")
            }
        }

        btnEnable.setOnClickListener {
            if (!isYoloReady) {
                showToast("AI 模型还没准备好")
                return@setOnClickListener
            }

            if (isControlEnabled) {
                showToast("追踪已经在运行")
                return@setOnClickListener
            }

            resetTrackingSession()

            if (!isGimbalDown) {
                showToast("正在调整云台...")
                controlGimbal(-90.0)
                isGimbalDown = true
                btnGimbal.text = "云台回正"
                mainView.postDelayed({ startVirtualStick() }, 1200)
            } else {
                startVirtualStick()
            }
        }

        btnStop.setOnClickListener {
            stopTracking("紧急停止", TrackingState.EMERGENCY_STOP)
        }
    }

    private fun startVirtualStick() {
        try {
            // 保证继续使用普通摇杆模式
            VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false)
        } catch (e: Exception) {
            Log.w(TAG, "setVirtualStickAdvancedModeEnabled(false) failed: ${e.message}")
        }

        VirtualStickManager.getInstance().enableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                isControlEnabled = true
                pidX.reset()
                pidY.reset()
                setTrackingState(TrackingState.SEARCHING, "已获取控制权，开始搜索目标")

                runOnUiThread {
                    btnEnable.text = "运行中..."
                    btnEnable.isEnabled = false
                }
            }

            override fun onFailure(error: IDJIError) {
                isControlEnabled = false
                setTrackingState(TrackingState.IDLE, "控制权获取失败: ${error.description()}")
            }
        })
    }

    private fun stopTracking(reason: String, nextState: TrackingState = TrackingState.EMERGENCY_STOP) {
        sendVelocityCommand(0.0, 0.0, 0.0, 0.0)

        isControlEnabled = false
        pidX.reset()
        pidY.reset()

        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0

        try {
            VirtualStickManager.getInstance().disableVirtualStick(null)
        } catch (e: Exception) {
            Log.w(TAG, "disableVirtualStick failed: ${e.message}")
        }

        setTrackingState(nextState, reason)

        runOnUiThread {
            btnEnable.text = "开启追踪"
            btnEnable.isEnabled = true
        }
    }

    private fun resetTrackingSession() {
        pidX.reset()
        pidY.reset()

        frameCounter = 0
        lastTargetRect = null
        trackingLostCount = 0
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0

        currentDisplayRect = null
        currentDisplayLabel = null
        currentDisplayConfidence = 0f
        currentSearchMode = SearchMode.GLOBAL
        currentRoiRect = null

        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdYaw = 0.0
        lastCmdVert = 0.0
        lastBoxRatio = 0.0
        lastErrX = 0.0
        lastErrY = 0.0

        setTrackingState(TrackingState.IDLE, "已重置会话")
        updateDashboard(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "待机")
    }

    private fun controlGimbal(pitch: Double) {
        val key = KeyTools.createKey(GimbalKey.KeyRotateByAngle)
        val rotation = GimbalAngleRotation().apply {
            mode = GimbalAngleRotationMode.ABSOLUTE_ANGLE
            this.pitch = pitch
        }
        KeyManager.getInstance().performAction(key, rotation, null)
    }

    private fun initDJIStream() {
        MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
            ComponentIndexType.LEFT_OR_MAIN,
            ICameraStreamManager.FrameFormat.NV21,
            frameListener
        )
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
            if (data.isEmpty() || width == 0 || height == 0) return
            if (!isYoloReady) return
            if (isProcessing.getAndSet(true)) return

            try {
                processFrame(data, width, height)
            } catch (e: Exception) {
                Log.e(TAG, "processFrame error: ${e.message}", e)
            } finally {
                isProcessing.set(false)
            }
        }
    }

    private fun processFrame(yuvData: ByteArray, width: Int, height: Int) {
        val yuvHeight = height + height / 2

        if (yuvMat == null || yuvMat?.width() != width || yuvMat?.height() != yuvHeight) {
            yuvMat?.release()
            rgbMat?.release()
            frame720p?.release()
            globalDetectMat?.release()

            yuvMat = Mat(yuvHeight, width, CvType.CV_8UC1)
            rgbMat = Mat(height, width, CvType.CV_8UC3)
            frame720p = Mat()
            globalDetectMat = Mat()
        }

        yuvMat!!.put(0, 0, yuvData)
        Imgproc.cvtColor(yuvMat!!, rgbMat!!, Imgproc.COLOR_YUV2RGB_NV21)
        Imgproc.resize(rgbMat!!, frame720p!!, Size(PROC_W.toDouble(), PROC_H.toDouble()))

        frameCounter++

        if (frameCounter % DETECT_INTERVAL == 0) {
            runDetectionPipeline(PROC_W, PROC_H)
        }

        drawVisionOverlay()

        val bitmap = Bitmap.createBitmap(frame720p!!.cols(), frame720p!!.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(frame720p!!, bitmap)
        runOnUiThread {
            mainView.setImageBitmap(bitmap)
        }
    }

    private fun runDetectionPipeline(imgW: Int, imgH: Int) {
        var results: List<DetectionResult> = emptyList()
        var roiForThisFrame: Rect? = null
        var searchModeForThisFrame = SearchMode.GLOBAL

        // A. 动态 ROI 追踪
        if (lastTargetRect != null && trackingLostCount < LOST_FRAME_THRESHOLD) {
            val roiRect = buildCropRect(lastTargetRect!!, imgW, imgH)
            val roiMat = Mat(frame720p!!, roiRect)
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

        // B. ROI 失败后尝试中心鹰眼
        if (results.isEmpty()) {
            val centerRect = Rect((imgW - CROP_SIZE) / 2, (imgH - CROP_SIZE) / 2, CROP_SIZE, CROP_SIZE)
            val centerMat = Mat(frame720p!!, centerRect)
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

        // C. 全图缩放搜索
        if (results.isEmpty()) {
            Imgproc.resize(frame720p!!, globalDetectMat!!, Size(CROP_SIZE.toDouble(), CROP_SIZE.toDouble()))
            results = yoloDetector.detect(globalDetectMat!!)
            roiForThisFrame = null
            searchModeForThisFrame = SearchMode.GLOBAL

            if (results.isNotEmpty()) {
                trackingLostCount = 0
            }
        }

        currentRoiRect = roiForThisFrame
        currentSearchMode = searchModeForThisFrame

        val target = results.firstOrNull { it.label in controllableLabels }

        if (target != null) {
            val displayRect = restoreRectTo720p(target.rect, roiForThisFrame, imgW, imgH)
            currentDisplayRect = Rect(displayRect.x, displayRect.y, displayRect.width, displayRect.height)
            currentDisplayLabel = target.label
            currentDisplayConfidence = target.confidence
            lastTargetRect = Rect(displayRect.x, displayRect.y, displayRect.width, displayRect.height)

            handleTargetFound(displayRect, imgW, imgH)
        } else {
            handleTargetLost()
        }
    }

    private fun handleTargetFound(rect: Rect, imgW: Int, imgH: Int) {
        val centerX = rect.x + rect.width / 2.0
        val centerY = rect.y + rect.height / 2.0

        val normErrorX = (centerX - imgW / 2.0) / (imgW / 2.0)
        val normErrorY = (centerY - imgH / 2.0) / (imgH / 2.0)
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

        if (trackingState == TrackingState.SEARCHING && consecutiveFoundFrames >= FOUND_FRAME_THRESHOLD) {
            setTrackingState(TrackingState.ALIGNING, "目标稳定，开始对准")
        }

        if (trackingState == TrackingState.ALIGNING) {
            if (alignedNow) {
                consecutiveAlignedFrames++
                if (consecutiveAlignedFrames >= ALIGN_FRAME_THRESHOLD) {
                    setTrackingState(TrackingState.DESCENDING, "对准稳定，开始下降")
                }
            } else {
                consecutiveAlignedFrames = 0
            }
        } else if (trackingState == TrackingState.DESCENDING) {
            if (!alignedNow) {
                consecutiveAlignedFrames = 0
                setTrackingState(TrackingState.ALIGNING, "偏差变大，回到对准")
            }
        }

        val vRoll = pidX.calculate(normErrorX * 2.0)
        val vPitch = pidY.calculate(-normErrorY * 2.0)
        val vVert = when (trackingState) {
            TrackingState.DESCENDING -> computeVerticalSpeed(boxRatio)
            else -> 0.0
        }

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
                updateDashboard(boxRatio, normErrorX, normErrorY, vRoll, vPitch, vVert, "对准后下降")
            }

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
        consecutiveLostFrames++

        if (consecutiveLostFrames > LOST_FRAME_THRESHOLD) {
            trackingLostCount = LOST_FRAME_THRESHOLD
            lastTargetRect = null
            currentDisplayRect = null
            currentDisplayLabel = null
            currentDisplayConfidence = 0f
            currentRoiRect = null
            currentSearchMode = SearchMode.GLOBAL

            if (isControlEnabled) {
                setTrackingState(TrackingState.LOST, "目标丢失")
            }
        } else {
            if (isControlEnabled && trackingState != TrackingState.SEARCHING) {
                setTrackingState(TrackingState.SEARCHING, "短时丢失，继续搜索")
            }
        }

        if (isControlEnabled) {
            sendVelocityCommand(0.0, 0.0, 0.0, 0.0)
        }

        updateDashboard(lastBoxRatio, 0.0, 0.0, 0.0, 0.0, 0.0, "Searching...")
    }

    private fun drawVisionOverlay() {
        when (currentSearchMode) {
            SearchMode.ROI -> {
                currentRoiRect?.let {
                    Imgproc.rectangle(frame720p!!, it, Scalar(255.0, 255.0, 0.0), 2)
                }
                Imgproc.putText(
                    frame720p!!,
                    "ROI TRACKING",
                    Point(20.0, 50.0),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar(255.0, 255.0, 0.0),
                    2
                )
            }

            SearchMode.EAGLE_EYE -> {
                currentRoiRect?.let {
                    Imgproc.rectangle(frame720p!!, it, Scalar(255.0, 255.0, 0.0), 2)
                }
                Imgproc.putText(
                    frame720p!!,
                    "EAGLE EYE",
                    Point(20.0, 50.0),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar(255.0, 255.0, 0.0),
                    2
                )
            }

            SearchMode.GLOBAL -> {
                Imgproc.putText(
                    frame720p!!,
                    "GLOBAL SEARCH",
                    Point(20.0, 50.0),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar(0.0, 255.0, 0.0),
                    2
                )
            }
        }

        currentDisplayRect?.let { rect ->
            Imgproc.rectangle(frame720p!!, rect, Scalar(0.0, 255.0, 0.0), 3)
            Imgproc.putText(
                frame720p!!,
                "${currentDisplayLabel ?: "target"} ${"%.2f".format(currentDisplayConfidence)}",
                Point(rect.x.toDouble(), rect.y - 10.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0.0, 255.0, 255.0),
                2
            )
        }
    }

    private fun buildCropRect(lastRect: Rect, imgW: Int, imgH: Int): Rect {
        val cx = lastRect.x + lastRect.width / 2
        val cy = lastRect.y + lastRect.height / 2

        val cropX = (cx - CROP_SIZE / 2).coerceIn(0, imgW - CROP_SIZE)
        val cropY = (cy - CROP_SIZE / 2).coerceIn(0, imgH - CROP_SIZE)

        return Rect(cropX, cropY, CROP_SIZE, CROP_SIZE)
    }

    private fun restoreRectTo720p(rawRect: Rect, roi: Rect?, imgW: Int, imgH: Int): Rect {
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
        showStatus("状态: ${stateLabel(newState)} | $detail")
    }

    private fun stateLabel(state: TrackingState): String {
        return when (state) {
            TrackingState.IDLE -> "IDLE"
            TrackingState.SEARCHING -> "SEARCHING"
            TrackingState.ALIGNING -> "ALIGNING"
            TrackingState.DESCENDING -> "DESCENDING"
            TrackingState.LOST -> "LOST"
            TrackingState.EMERGENCY_STOP -> "EMERGENCY_STOP"
        }
    }

    private fun updateDashboard(
        ratio: Double,
        errX: Double,
        errY: Double,
        cmdRoll: Double,
        cmdPitch: Double,
        cmdVert: Double,
        detail: String
    ) {
        runOnUiThread {
            tvHeight.text = "目标占比: %.2f".format(ratio)
            tvErrorX.text = "ErrX: %.3f".format(errX)
            tvErrorY.text = "ErrY: %.3f".format(errY)
            tvCmd.text = "Cmd R/P/V: %.2f / %.2f / %.2f".format(cmdRoll, cmdPitch, cmdVert)
            tvStatus.text = "状态: ${stateLabel(trackingState)} | $detail"
        }
    }

    private fun updateVsStateText() {
        runOnUiThread {
            tvVsState.text = "VS: ${if (lastVsEnabled) "ON" else "OFF"} | Authority: $lastVsAuthority | Reason: $lastVsReason"
        }
    }

    private fun showStatus(text: String) {
        runOnUiThread {
            tvStatus.text = text
        }
    }

    private fun showToast(msg: String) {
        runOnUiThread {
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        stopTracking("页面关闭，释放控制", TrackingState.IDLE)

        try {
            MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(frameListener)
        } catch (e: Exception) {
            Log.w(TAG, "removeFrameListener failed: ${e.message}")
        }

        try {
            VirtualStickManager.getInstance().removeVirtualStickStateListener(vsStateListener)
        } catch (e: Exception) {
            Log.w(TAG, "removeVirtualStickStateListener failed: ${e.message}")
        }

        yuvMat?.release()
        rgbMat?.release()
        frame720p?.release()
        globalDetectMat?.release()

        super.onDestroy()
    }
}