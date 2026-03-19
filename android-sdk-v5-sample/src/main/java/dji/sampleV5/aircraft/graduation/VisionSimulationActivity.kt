package dji.sampleV5.aircraft.graduation

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import dji.v5.manager.SDKManager
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.max

class VisionSimulationActivity : AppCompatActivity() {

    private enum class TrackingState {
        IDLE,
        SEARCHING,
        ALIGNING,
        DESCENDING,
        AUTO_LANDING,
        LOST
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

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var loadingBar: ProgressBar
    private lateinit var switchVideoBtn: Button
    private lateinit var controlBtn: Button

    @Volatile
    private var isRunning = false

    @Volatile
    private var isYoloReady = false

    private val videoResources = listOf(
        "sim_01_vertical",
        "sim_02_horizontal",
        "sim_03_rotate"
    )
    private var currentVideoIndex = 0

    private lateinit var yoloDetector: YoloV8Detector

    private var runMode = RunMode.PREVIEW
    private var trackingState = TrackingState.IDLE
    private var autoLandingActive = false

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

    private var lastBoxRatio = 0.0
    private var lastErrX = 0.0
    private var lastErrY = 0.0
    private var lastCmdRoll = 0.0
    private var lastCmdPitch = 0.0
    private var lastCmdVert = 0.0
    private var lastStatusDetail = "待机"

    private var previewFrameCount = 0
    private var detectFrameCount = 0
    private var previewFps = 0.0
    private var detectFps = 0.0
    private var fpsWindowStartMs = 0L
    private var lastDetectRequestMs = 0L

    companion object {
        private const val TAG = "VisionSimulation"

        private const val PROC_W = 640
        private const val PROC_H = 360
        private const val CROP_SIZE = 320
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

        private const val TARGET_FRAME_INTERVAL_MS = 33L
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupUserInterface()
        verifySdkStatus()

        if (OpenCVLoader.initDebug()) {
            appendStatus("OpenCV 初始化成功，正在加载模型...")
            initYoloModel()
        } else {
            appendStatus("严重错误: OpenCV 初始化失败")
        }
    }

    private fun initYoloModel() {
        yoloDetector = YoloV8Detector()
        Thread {
            try {
                val modelPath = AssetUtils.getAssetFilePath(this, "yolov8n.onnx")
                if (yoloDetector.init(modelPath)) {
                    isYoloReady = true
                    runOnUiThread {
                        appendStatus("YOLO 模型加载成功")
                        playCurrentVideo()
                    }
                } else {
                    runOnUiThread {
                        appendStatus("模型初始化失败")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "initYoloModel error: ${e.message}", e)
                runOnUiThread {
                    appendStatus("模型加载异常: ${e.message}")
                }
            }
        }.start()
    }

    private fun playCurrentVideo() {
        if (!isYoloReady) return

        isRunning = false
        runOnUiThread {
            loadingBar.visibility = View.VISIBLE
            switchVideoBtn.isEnabled = false
            controlBtn.isEnabled = false
            switchVideoBtn.text = "加载中..."
        }

        Thread {
            Thread.sleep(300)

            val videoName = videoResources[currentVideoIndex]
            val videoPath = copyRawResourceToCache(videoName)
            if (videoPath == null) {
                runOnUiThread { appendStatus("文件不存在: $videoName") }
                return@Thread
            }

            var capture = VideoCapture(videoPath)
            if (!capture.isOpened) {
                runOnUiThread { appendStatus("无法打开视频流: $videoName") }
                return@Thread
            }

            runOnUiThread {
                loadingBar.visibility = View.GONE
                switchVideoBtn.isEnabled = true
                controlBtn.isEnabled = true
                switchVideoBtn.text = "切换素材 (${currentVideoIndex + 1}/${videoResources.size})"
                updateControlButtonUi()
            }

            isRunning = true
            resetTrackingState()
            fpsWindowStartMs = SystemClock.uptimeMillis()

            val frameSrc = Mat()
            val frameProc = Mat(PROC_H, PROC_W, CvType.CV_8UC3)
            val globalDetectMat = Mat(CROP_SIZE, CROP_SIZE, CvType.CV_8UC3)
            val displayBitmap = Bitmap.createBitmap(PROC_W, PROC_H, Bitmap.Config.ARGB_8888)

            while (isRunning) {
                val loopStart = SystemClock.uptimeMillis()

                if (!capture.read(frameSrc)) {
                    capture.release()
                    capture = VideoCapture(videoPath)
                    resetTrackingState()
                    continue
                }

                if (frameSrc.empty()) continue

                try {
                    Imgproc.resize(frameSrc, frameProc, Size(PROC_W.toDouble(), PROC_H.toDouble()))

                    val now = SystemClock.uptimeMillis()
                    if (now - lastDetectRequestMs >= DETECT_INTERVAL_MS) {
                        lastDetectRequestMs = now
                        runDetectionPipeline(frameProc, globalDetectMat, PROC_W, PROC_H)
                        markDetectFrame()
                    }

                    drawVisionOverlay(frameProc)
                    Utils.matToBitmap(frameProc, displayBitmap)
                    markPreviewFrame()

                    runOnUiThread {
                        imageView.setImageBitmap(displayBitmap)
                        statusText.text = buildStatusText(videoName)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "loop error: ${e.message}", e)
                }

                val processTime = SystemClock.uptimeMillis() - loopStart
                val sleepTime = TARGET_FRAME_INTERVAL_MS - processTime
                if (sleepTime > 0) Thread.sleep(sleepTime)
            }

            capture.release()
            frameSrc.release()
            frameProc.release()
            globalDetectMat.release()
        }.start()
    }

    private fun runDetectionPipeline(frame: Mat, globalDetectMat: Mat, imgW: Int, imgH: Int) {
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
            Imgproc.resize(frame, globalDetectMat, Size(CROP_SIZE.toDouble(), CROP_SIZE.toDouble()))
            results = yoloDetector.detect(globalDetectMat)
            roiForThisFrame = null
            searchModeForThisFrame = SearchMode.GLOBAL

            if (results.isNotEmpty()) {
                trackingLostCount = 0
            }
        }

        currentRoiRect = roiForThisFrame
        currentSearchMode = searchModeForThisFrame

        val allTargets = results.map { result ->
            val restored = restoreRectToProc(result.rect, roiForThisFrame, imgW, imgH)
            OverlayTarget(
                label = result.label,
                confidence = result.confidence,
                rect = restored
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
        lastCmdVert = 0.0
        if (!shouldKeepCurrentStateInPreview()) {
            trackingState = TrackingState.IDLE
            lastStatusDetail = "预览模式：仅检测显示"
        }
    }

    private fun holdPreviewNoTarget() {
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastErrX = 0.0
        lastErrY = 0.0
        if (!shouldKeepCurrentStateInPreview()) {
            trackingState = TrackingState.IDLE
            lastStatusDetail = "预览模式：未发现目标"
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
            trackingState = TrackingState.SEARCHING
            lastStatusDetail = "重新发现目标，等待稳定"
        }

        val alignedNow = abs(normErrorX) < ALIGN_THRESHOLD_X && abs(normErrorY) < ALIGN_THRESHOLD_Y
        val finalAlignedNow = abs(normErrorX) < FINAL_ALIGN_THRESHOLD_X && abs(normErrorY) < FINAL_ALIGN_THRESHOLD_Y

        if (trackingState == TrackingState.SEARCHING && consecutiveFoundFrames >= FOUND_FRAME_THRESHOLD) {
            trackingState = TrackingState.ALIGNING
            lastStatusDetail = "目标稳定，开始对准"
        }

        if (trackingState == TrackingState.ALIGNING) {
            if (alignedNow) {
                consecutiveAlignedFrames++
                if (consecutiveAlignedFrames >= ALIGN_FRAME_THRESHOLD) {
                    trackingState = TrackingState.DESCENDING
                    consecutiveFinalFrames = 0
                    lastStatusDetail = "对准稳定，开始下降"
                }
            } else {
                consecutiveAlignedFrames = 0
            }
        } else if (trackingState == TrackingState.DESCENDING) {
            if (!alignedNow) {
                consecutiveAlignedFrames = 0
                consecutiveFinalFrames = 0
                trackingState = TrackingState.ALIGNING
                lastStatusDetail = "偏差变大，回到对准"
            } else if (finalAlignedNow && boxRatio >= FINAL_BOX_RATIO_THRESHOLD) {
                consecutiveFinalFrames++
                if (consecutiveFinalFrames >= FINAL_FRAME_THRESHOLD) {
                    startAutoLandingTransfer("进入最终阶段，切自动降落")
                    return
                }
            } else {
                consecutiveFinalFrames = 0
            }
        }

        val vRoll = normErrorX * 0.5
        val vPitch = -normErrorY * 0.5
        val vVert = if (trackingState == TrackingState.DESCENDING) computeVerticalSpeed(boxRatio) else 0.0

        when (trackingState) {
            TrackingState.SEARCHING -> {
                lastCmdRoll = 0.0
                lastCmdPitch = 0.0
                lastCmdVert = 0.0
                lastStatusDetail = "已发现目标，等待稳定"
            }

            TrackingState.ALIGNING -> {
                lastCmdRoll = vRoll
                lastCmdPitch = vPitch
                lastCmdVert = 0.0
                lastStatusDetail = "正在对准"
            }

            TrackingState.DESCENDING -> {
                lastCmdRoll = vRoll
                lastCmdPitch = vPitch
                lastCmdVert = vVert
                lastStatusDetail = "视觉下降中"
            }

            TrackingState.AUTO_LANDING,
            TrackingState.IDLE,
            TrackingState.LOST -> {
                lastCmdRoll = 0.0
                lastCmdPitch = 0.0
                lastCmdVert = 0.0
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
            trackingState = TrackingState.LOST
            lastStatusDetail = "目标丢失"
        } else if (trackingState != TrackingState.IDLE) {
            trackingState = TrackingState.SEARCHING
            lastStatusDetail = "短时丢失，继续搜索"
        }

        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastErrX = 0.0
        lastErrY = 0.0
    }

    private fun startAutoLandingTransfer(detail: String) {
        autoLandingActive = true
        runMode = RunMode.PREVIEW
        trackingState = TrackingState.AUTO_LANDING
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveFinalFrames = 0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastStatusDetail = detail
        runOnUiThread { updateControlButtonUi() }
    }

    private fun enterControlMode() {
        autoLandingActive = false
        runMode = RunMode.CONTROL
        trackingState = TrackingState.SEARCHING
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastStatusDetail = "控制模式已开启"
        updateControlButtonUi()
    }

    private fun exitControlMode() {
        runMode = RunMode.PREVIEW
        autoLandingActive = false
        trackingState = TrackingState.IDLE
        consecutiveFoundFrames = 0
        consecutiveAlignedFrames = 0
        consecutiveLostFrames = 0
        consecutiveFinalFrames = 0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastStatusDetail = "已退出控制模式"
        updateControlButtonUi()
    }

    private fun drawVisionOverlay(frame: Mat) {
        currentRoiRect?.let {
            Imgproc.rectangle(frame, it, Scalar(255.0, 255.0, 0.0), 2)
        }

        Imgproc.putText(
            frame,
            buildOverlayModeText(),
            Point(20.0, 35.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(255.0, 255.0, 0.0),
            2
        )

        currentDisplayTargets.filter { !it.isMainTarget }.forEach { target ->
            Imgproc.rectangle(frame, target.rect, Scalar(255.0, 255.0, 0.0), 2)
            Imgproc.putText(
                frame,
                "${target.label} ${"%.2f".format(target.confidence)}",
                Point(target.rect.x.toDouble(), max(20, target.rect.y - 8).toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(255.0, 255.0, 0.0),
                2
            )
        }

        currentDisplayTargets.firstOrNull { it.isMainTarget }?.let { target ->
            Imgproc.rectangle(frame, target.rect, Scalar(0.0, 255.0, 0.0), 3)
            Imgproc.putText(
                frame,
                "MAIN ${target.label} ${"%.2f".format(target.confidence)}",
                Point(target.rect.x.toDouble(), max(20, target.rect.y - 8).toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(0.0, 255.0, 0.0),
                2
            )
        }
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

    private fun stateLabel(state: TrackingState): String {
        return when (state) {
            TrackingState.IDLE -> "IDLE"
            TrackingState.SEARCHING -> "SEARCHING"
            TrackingState.ALIGNING -> "ALIGNING"
            TrackingState.DESCENDING -> "DESCENDING"
            TrackingState.AUTO_LANDING -> "AUTO_LANDING"
            TrackingState.LOST -> "LOST"
        }
    }

    private fun buildStatusText(videoName: String): String {
        val mainTargetText = currentMainTarget?.let {
            "${it.label} ${"%.2f".format(it.confidence)}"
        } ?: "None"

        return buildString {
            append("Video: ")
            append(videoName)
            append("\n")
            append("FPS: ")
            append(String.format("%.1f", previewFps))
            append(" | DET: ")
            append(String.format("%.1f", detectFps))
            append("Hz")
            append("\n")
            append("RunMode: ")
            append(runMode.name)
            append(" | State: ")
            append(stateLabel(trackingState))
            append("\n")
            append("Mode: ")
            append(currentSearchMode.name)
            append(" | Boxes: ")
            append(currentDisplayTargets.size)
            append("\n")
            append("Main: ")
            append(mainTargetText)
            append(" | Ratio: ")
            append(String.format("%.2f", lastBoxRatio))
            append("\n")
            append("ErrX: ")
            append(String.format("%.3f", lastErrX))
            append(" | ErrY: ")
            append(String.format("%.3f", lastErrY))
            append("\n")
            append("Cmd R/P/V: ")
            append(String.format("%.2f / %.2f / %.2f", lastCmdRoll, lastCmdPitch, lastCmdVert))
            append("\n")
            append("Detail: ")
            append(lastStatusDetail)
        }
    }

    private fun markPreviewFrame() {
        previewFrameCount++
        updateFpsWindow()
    }

    private fun markDetectFrame() {
        detectFrameCount++
        updateFpsWindow()
    }

    private fun updateFpsWindow() {
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

    private fun resetTrackingState() {
        runMode = RunMode.PREVIEW
        trackingState = TrackingState.IDLE
        autoLandingActive = false

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

        lastBoxRatio = 0.0
        lastErrX = 0.0
        lastErrY = 0.0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
        lastStatusDetail = "待机"

        lastDetectRequestMs = 0L
        previewFrameCount = 0
        detectFrameCount = 0
        previewFps = 0.0
        detectFps = 0.0
        fpsWindowStartMs = SystemClock.uptimeMillis()

        runOnUiThread { updateControlButtonUi() }
    }

    private fun onSwitchVideoClicked() {
        currentVideoIndex++
        if (currentVideoIndex >= videoResources.size) {
            currentVideoIndex = 0
        }
        playCurrentVideo()
    }

    private fun setupUserInterface() {
        val rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.BLACK)

        imageView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        rootLayout.addView(
            imageView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        statusText = TextView(this).apply {
            text = "初始化中..."
            setTextColor(Color.GREEN)
            textSize = 15f
            setPadding(20, 20, 20, 20)
            setBackgroundColor(Color.parseColor("#66000000"))
        }
        rootLayout.addView(statusText)

        loadingBar = ProgressBar(this)
        val paramsBar = FrameLayout.LayoutParams(100, 100)
        paramsBar.gravity = Gravity.CENTER
        loadingBar.visibility = View.GONE
        rootLayout.addView(loadingBar, paramsBar)

        val controlPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val panelParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        panelParams.gravity = Gravity.BOTTOM or Gravity.END
        panelParams.setMargins(0, 0, 50, 50)

        switchVideoBtn = Button(this).apply {
            text = "切换视频素材"
            setOnClickListener { onSwitchVideoClicked() }
        }
        controlPanel.addView(switchVideoBtn)

        controlBtn = Button(this).apply {
            text = "进入控制模式"
            setOnClickListener {
                if (runMode == RunMode.PREVIEW) enterControlMode() else exitControlMode()
            }
        }
        val controlBtnParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        controlBtnParams.topMargin = 20
        controlPanel.addView(controlBtn, controlBtnParams)

        rootLayout.addView(controlPanel, panelParams)

        setContentView(rootLayout)
    }

    private fun updateControlButtonUi() {
        if (!::controlBtn.isInitialized) return
        when {
            autoLandingActive -> {
                controlBtn.text = "自动降落接管中"
                controlBtn.isEnabled = false
            }
            runMode == RunMode.CONTROL -> {
                controlBtn.text = "退出控制模式"
                controlBtn.isEnabled = true
            }
            else -> {
                controlBtn.text = "进入控制模式"
                controlBtn.isEnabled = true
            }
        }
    }

    private fun shouldKeepCurrentStateInPreview(): Boolean {
        return autoLandingActive || trackingState == TrackingState.AUTO_LANDING
    }

    private fun isSameRect(a: Rect, b: Rect): Boolean {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height
    }

    private fun copyRawResourceToCache(resName: String): String? {
        return try {
            val resId = resources.getIdentifier(resName, "raw", packageName)
            if (resId == 0) return null

            val file = File(cacheDir, "$resName.mp4")
            if (file.exists()) {
                file.delete()
            }

            resources.openRawResource(resId).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            file.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "copyRawResourceToCache error: ${e.message}", e)
            null
        }
    }

    private fun verifySdkStatus() {
        try {
            SDKManager.getInstance().sdkVersion
        } catch (_: Exception) {
        }
    }

    private fun appendStatus(text: String) {
        runOnUiThread {
            statusText.text = text
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isRunning = false
    }
}
