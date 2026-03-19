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
import kotlin.math.min

class VisionSimulationActivity : AppCompatActivity() {

    private enum class TrackingState {
        IDLE,
        SEARCHING,
        ALIGNING,
        DESCENDING,
        LOST
    }

    private enum class SearchMode {
        GLOBAL,
        EAGLE_EYE,
        ROI
    }

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var loadingBar: ProgressBar
    private lateinit var switchVideoBtn: Button

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

    // 跟实机逻辑对齐的状态变量
    private var trackingState = TrackingState.IDLE
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

    private var lastBoxRatio = 0.0
    private var lastErrX = 0.0
    private var lastErrY = 0.0
    private var lastCmdRoll = 0.0
    private var lastCmdPitch = 0.0
    private var lastCmdVert = 0.0

    // FPS / 检测频率
    private var previewFrameCount = 0
    private var detectFrameCount = 0
    private var previewFps = 0.0
    private var detectFps = 0.0
    private var fpsWindowStartMs = 0L
    private var lastDetectRequestMs = 0L

    private val controllableLabels = setOf(
        "land_h",
        "blue_circle_2",
        "green_circle_3",
        "red_circle_1"
    )

    companion object {
        private const val TAG = "VisionSimulation"

        // 默认先和你当前满意的实机检测分辨率保持一致
        private const val PROC_W = 640
        private const val PROC_H = 360
        private const val CROP_SIZE = 320

        // 先和你当前实机手动改成的 300ms 对齐
        private const val DETECT_INTERVAL_MS = 300L

        private const val LOST_FRAME_THRESHOLD = 10
        private const val FOUND_FRAME_THRESHOLD = 3
        private const val ALIGN_FRAME_THRESHOLD = 5

        private const val ALIGN_THRESHOLD_X = 0.08
        private const val ALIGN_THRESHOLD_Y = 0.08

        // 仿真预览按 30fps 节奏播放
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
                switchVideoBtn.text = "切换素材 (${currentVideoIndex + 1}/${videoResources.size})"
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
                    // 1) 统一处理分辨率
                    Imgproc.resize(frameSrc, frameProc, Size(PROC_W.toDouble(), PROC_H.toDouble()))

                    // 2) 低频检测
                    val now = SystemClock.uptimeMillis()
                    if (now - lastDetectRequestMs >= DETECT_INTERVAL_MS) {
                        lastDetectRequestMs = now
                        runDetectionPipeline(frameProc, globalDetectMat, PROC_W, PROC_H)
                        markDetectFrame()
                    }

                    // 3) 画叠加层
                    drawVisionOverlay(frameProc)

                    // 4) 刷新预览
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

        // A. 动态 ROI 追踪
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

        // B. 中心鹰眼
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

        // C. 全图缩放搜索
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

        val target = results.firstOrNull { it.label in controllableLabels }

        if (target != null) {
            val displayRect = restoreRectToProc(target.rect, roiForThisFrame, imgW, imgH)
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
            trackingState = TrackingState.SEARCHING
        }

        val alignedNow = abs(normErrorX) < ALIGN_THRESHOLD_X && abs(normErrorY) < ALIGN_THRESHOLD_Y

        if (trackingState == TrackingState.SEARCHING && consecutiveFoundFrames >= FOUND_FRAME_THRESHOLD) {
            trackingState = TrackingState.ALIGNING
        }

        if (trackingState == TrackingState.ALIGNING) {
            if (alignedNow) {
                consecutiveAlignedFrames++
                if (consecutiveAlignedFrames >= ALIGN_FRAME_THRESHOLD) {
                    trackingState = TrackingState.DESCENDING
                }
            } else {
                consecutiveAlignedFrames = 0
            }
        } else if (trackingState == TrackingState.DESCENDING) {
            if (!alignedNow) {
                consecutiveAlignedFrames = 0
                trackingState = TrackingState.ALIGNING
            }
        }

        val vRoll = normErrorX * 0.5
        val vPitch = -normErrorY * 0.5
        val vVert = when (trackingState) {
            TrackingState.DESCENDING -> computeVerticalSpeed(boxRatio)
            else -> 0.0
        }

        lastCmdRoll = vRoll
        lastCmdPitch = vPitch
        lastCmdVert = vVert
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
            trackingState = TrackingState.LOST
        } else {
            if (trackingState != TrackingState.IDLE) {
                trackingState = TrackingState.SEARCHING
            }
        }

        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0
    }

    private fun drawVisionOverlay(frame: Mat) {
        currentRoiRect?.let {
            Imgproc.rectangle(frame, it, Scalar(255.0, 255.0, 0.0), 2)
        }

        val modeText = when (currentSearchMode) {
            SearchMode.ROI -> "ROI TRACKING"
            SearchMode.EAGLE_EYE -> "EAGLE EYE"
            SearchMode.GLOBAL -> "GLOBAL SEARCH"
        }

        Imgproc.putText(
            frame,
            modeText,
            Point(20.0, 35.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(255.0, 255.0, 0.0),
            2
        )

        currentDisplayRect?.let { rect ->
            Imgproc.rectangle(frame, rect, Scalar(0.0, 255.0, 0.0), 2)
            val label = "${currentDisplayLabel ?: "target"} ${"%.2f".format(currentDisplayConfidence)}"
            Imgproc.putText(
                frame,
                label,
                Point(rect.x.toDouble(), max(20, rect.y - 8).toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0.0, 255.0, 255.0),
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

    private fun stateLabel(state: TrackingState): String {
        return when (state) {
            TrackingState.IDLE -> "IDLE"
            TrackingState.SEARCHING -> "SEARCHING"
            TrackingState.ALIGNING -> "ALIGNING"
            TrackingState.DESCENDING -> "DESCENDING"
            TrackingState.LOST -> "LOST"
        }
    }

    private fun modeLabel(mode: SearchMode): String {
        return when (mode) {
            SearchMode.ROI -> "ROI"
            SearchMode.EAGLE_EYE -> "EAGLE"
            SearchMode.GLOBAL -> "GLOBAL"
        }
    }

    private fun buildStatusText(videoName: String): String {
        val targetText = if (currentDisplayLabel != null) {
            "${currentDisplayLabel} ${"%.2f".format(currentDisplayConfidence)}"
        } else {
            "None"
        }

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
            append("State: ")
            append(stateLabel(trackingState))
            append(" | Mode: ")
            append(modeLabel(currentSearchMode))
            append("\n")
            append("Target: ")
            append(targetText)
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
        trackingState = TrackingState.IDLE
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

        lastBoxRatio = 0.0
        lastErrX = 0.0
        lastErrY = 0.0
        lastCmdRoll = 0.0
        lastCmdPitch = 0.0
        lastCmdVert = 0.0

        lastDetectRequestMs = 0L
        previewFrameCount = 0
        detectFrameCount = 0
        previewFps = 0.0
        detectFps = 0.0
        fpsWindowStartMs = SystemClock.uptimeMillis()
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

        val controlPanel = FrameLayout(this)
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

        rootLayout.addView(controlPanel, panelParams)

        setContentView(rootLayout)
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