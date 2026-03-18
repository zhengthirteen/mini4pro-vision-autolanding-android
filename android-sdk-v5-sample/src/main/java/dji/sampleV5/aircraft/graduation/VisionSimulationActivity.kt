package dji.sampleV5.aircraft.graduation

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import dji.v5.manager.SDKManager
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

class VisionSimulationActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var loadingBar: ProgressBar
    private lateinit var switchVideoBtn: Button
    private lateinit var colorFixBtn: Button

    @Volatile private var isRunning = false

    private val videoResources = listOf(
        "sim_01_vertical",
        "sim_02_horizontal",
        "sim_03_rotate"
    )
    private var currentVideoIndex = 0

    private lateinit var yoloDetector: YoloV8Detector
    @Volatile private var isYoloReady = false
    @Volatile private var needSwapRB = false

    // [新增] 动态追踪状态变量
    private var lastTargetRect: Rect? = null
    private var trackingLostCount = 0

    // 缓存结果用于绘制
    private var lastResults: List<DetectionResult> = emptyList()
    // 缓存当前模式：0=Global, 1=EagleEye(Center), 2=ROI(Dynamic)
    private var currentSearchMode = 0
    private var currentRoiRect: Rect? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupUserInterface()
        verifySdkStatus()

        if (OpenCVLoader.initDebug()) {
            appendStatus("OpenCV 初始化成功")
            initYoloModel()
        } else {
            appendStatus("严重错误: OpenCV 初始化失败!")
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
                        appendStatus("YOLO 模型加载成功！")
                        playCurrentVideo()
                    }
                } else {
                    runOnUiThread { appendStatus("模型初始化失败！") }
                }
            } catch (e: Exception) {
                runOnUiThread { appendStatus("模型加载异常: ${e.message}") }
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
            Thread.sleep(500)

            val videoName = videoResources[currentVideoIndex]
            val videoPath = copyRawResourceToCache(videoName)

            if (videoPath == null) {
                runOnUiThread { appendStatus("文件不存在: $videoName") }
                return@Thread
            }

            var capture = VideoCapture(videoPath)
            if (!capture.isOpened) {
                runOnUiThread { appendStatus("无法打开视频流") }
                return@Thread
            }

            // 仿真不加延时，全速运行测试性能
            val sleepTime = 15L

            runOnUiThread {
                loadingBar.visibility = View.GONE
                switchVideoBtn.isEnabled = true
                switchVideoBtn.text = "切换素材 (${currentVideoIndex + 1}/${videoResources.size})"
                appendStatus("播放: $videoName")
            }

            isRunning = true

            // 内存复用
            val frame4K = Mat()
            val frame720p = Mat()
            val frameRGB = Mat()
            val inputMat = Mat()

            val targetSize = Size(1280.0, 720.0)
            val procW = 1280
            val procH = 720

            var frameCounter = 0
            val DETECT_INTERVAL = 3

            // 重置追踪状态
            lastTargetRect = null
            trackingLostCount = 0

            while (isRunning) {
                val loopStart = System.currentTimeMillis()

                if (!capture.read(frame4K)) {
                    capture.release()
                    capture = VideoCapture(videoPath)
                    // 重置追踪状态
                    lastTargetRect = null
                    continue
                }

                if (frame4K.empty()) continue

                try {
                    // 1. 缩放至 720p
                    Imgproc.resize(frame4K, frame720p, targetSize)

                    // 2. 颜色转换
                    if (frame720p.channels() == 4) {
                        if (needSwapRB) Imgproc.cvtColor(frame720p, frameRGB, Imgproc.COLOR_BGRA2RGB)
                        else Imgproc.cvtColor(frame720p, frameRGB, Imgproc.COLOR_RGBA2RGB)
                    } else {
                        if (needSwapRB) Imgproc.cvtColor(frame720p, frameRGB, Imgproc.COLOR_BGR2RGB)
                        else frame720p.copyTo(frameRGB)
                    }

                    // ==========================================
                    // 核心视觉逻辑 (ROI -> Eagle -> Global)
                    // ==========================================
                    if (frameCounter % DETECT_INTERVAL == 0) {
                        var results: List<DetectionResult> = emptyList()
                        var roiRect: Rect? = null
                        var mode = 0 // 0:Global, 1:Eagle, 2:ROI

                        // --- 策略 A: [导师建议] 动态 ROI 追踪 ---
                        // 如果上一帧有目标，直接锁死在目标周围找
                        if (lastTargetRect != null && trackingLostCount < 10) {
                            val cx = lastTargetRect!!.x + lastTargetRect!!.width / 2
                            val cy = lastTargetRect!!.y + lastTargetRect!!.height / 2

                            // 限制边界
                            val cropX = (cx - 320).coerceIn(0, procW - 640)
                            val cropY = (cy - 320).coerceIn(0, procH - 640)

                            roiRect = Rect(cropX, cropY, 640, 640)
                            val cropMat = Mat(frameRGB, roiRect)

                            results = yoloDetector.detect(cropMat)

                            if (results.isNotEmpty()) {
                                mode = 2 // ROI Mode
                                trackingLostCount = 0
                            } else {
                                trackingLostCount++
                            }
                        }

                        // --- 策略 B: 搜索模式 (如果 ROI 失败) ---
                        if (results.isEmpty()) {
                            // 1. 尝试中心鹰眼 (High Altitude)
                            val centerX = (procW - 640) / 2
                            val centerY = (procH - 640) / 2
                            val centerRect = Rect(centerX, centerY, 640, 640)
                            val centerMat = Mat(frameRGB, centerRect)
                            results = yoloDetector.detect(centerMat)

                            if (results.isNotEmpty()) {
                                mode = 1 // Eagle Eye Mode
                                roiRect = centerRect
                                trackingLostCount = 0
                            } else {
                                // 2. 尝试全图缩放 (Low Altitude)
                                Imgproc.resize(frameRGB, inputMat, Size(640.0, 640.0))
                                results = yoloDetector.detect(inputMat)
                                mode = 0 // Global Mode
                                roiRect = null

                                if (results.isNotEmpty()) trackingLostCount = 0
                            }
                        }

                        // 更新全局状态供绘制
                        lastResults = results
                        currentSearchMode = mode
                        currentRoiRect = roiRect

                        // 更新 lastTargetRect 供下一帧使用
                        if (results.isNotEmpty()) {
                            val rawRes = results[0]
                            val realRect: Rect
                            // 坐标还原逻辑
                            if (roiRect != null) {
                                // 加法还原
                                realRect = Rect(
                                    rawRes.rect.x + roiRect.x,
                                    rawRes.rect.y + roiRect.y,
                                    rawRes.rect.width,
                                    rawRes.rect.height
                                )
                            } else {
                                // 乘法还原
                                val scaleX = procW.toDouble() / 640.0
                                val scaleY = procH.toDouble() / 640.0
                                realRect = Rect(
                                    (rawRes.rect.x * scaleX).toInt(),
                                    (rawRes.rect.y * scaleY).toInt(),
                                    (rawRes.rect.width * scaleX).toInt(),
                                    (rawRes.rect.height * scaleY).toInt()
                                )
                            }
                            lastTargetRect = realRect
                        } else if (trackingLostCount > 10) {
                            lastTargetRect = null
                        }
                    }
                    frameCounter++

                    // ==========================================
                    // 绘制可视化结果
                    // ==========================================

                    // 1. 绘制 ROI 框 (显示我们在看哪里)
                    if (currentRoiRect != null) {
                        Imgproc.rectangle(frameRGB, currentRoiRect, Scalar(255.0, 255.0, 0.0), 2)
                    }

                    // 2. 显示当前模式文字
                    val modeText = when(currentSearchMode) {
                        2 -> "ROI TRACKING (FAST)"  // 导师建议的模式
                        1 -> "EAGLE EYE (CENTER)"
                        else -> "GLOBAL SEARCH"
                    }
                    val textColor = if (currentSearchMode == 2) Scalar(0.0, 255.0, 0.0) else Scalar(0.0, 255.0, 255.0)
                    Imgproc.putText(frameRGB, modeText, Point(20.0, 50.0),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2)

                    // 3. 绘制最终目标框
                    for (res in lastResults) {
                        val displayRect: Rect
                        if (currentRoiRect != null) {
                            displayRect = Rect(
                                res.rect.x + currentRoiRect!!.x,
                                res.rect.y + currentRoiRect!!.y,
                                res.rect.width,
                                res.rect.height
                            )
                        } else {
                            val scaleX = procW.toDouble() / 640.0
                            val scaleY = procH.toDouble() / 640.0
                            displayRect = Rect(
                                (res.rect.x * scaleX).toInt(),
                                (res.rect.y * scaleY).toInt(),
                                (res.rect.width * scaleX).toInt(),
                                (res.rect.height * scaleY).toInt()
                            )
                        }

                        Imgproc.rectangle(frameRGB, displayRect, Scalar(0.0, 255.0, 0.0), 3)
                        val label = "${res.label} ${(res.confidence * 100).toInt()}%"
                        Imgproc.putText(frameRGB, label, Point(displayRect.x.toDouble(), displayRect.y - 10.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0.0, 255.0, 255.0), 2)
                    }

                    // 显示
                    val finalBitmap = Bitmap.createBitmap(procW, procH, Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(frameRGB, finalBitmap)

                    runOnUiThread {
                        imageView.setImageBitmap(finalBitmap)
                        if (frameCounter % 10 == 0) {
                            statusText.text = if (lastResults.isNotEmpty())
                                "Target: ${lastResults[0].label}" else "Searching..."
                        }
                    }

                } catch (e: Exception) {
                    Log.e("VisionSim", "Error: ${e.message}")
                }

                val processTime = System.currentTimeMillis() - loopStart
                val wait = sleepTime - processTime
                if (wait > 0) Thread.sleep(wait)
            }

            capture.release()
            frame4K.release()
            frame720p.release()
            frameRGB.release()
            inputMat.release()
        }.start()
    }

    private fun onSwitchVideoClicked() {
        currentVideoIndex++
        if (currentVideoIndex >= videoResources.size) {
            currentVideoIndex = 0
        }
        playCurrentVideo()
    }

    private fun onColorFixClicked() {
        needSwapRB = !needSwapRB
        val mode = if (needSwapRB) "交换 RB" else "直通 RGB"
        colorFixBtn.text = "颜色修正: $mode"
        runOnUiThread { Toast.makeText(this, "模式已切换为: $mode", Toast.LENGTH_SHORT).show() }
    }

    private fun setupUserInterface() {
        val rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.BLACK)

        imageView = ImageView(this)
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER
        rootLayout.addView(imageView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))

        statusText = TextView(this)
        statusText.text = "初始化中..."
        statusText.setTextColor(Color.GREEN)
        statusText.textSize = 16f
        statusText.setPadding(20, 20, 20, 20)
        rootLayout.addView(statusText)

        loadingBar = ProgressBar(this)
        val paramsBar = FrameLayout.LayoutParams(100, 100)
        paramsBar.gravity = Gravity.CENTER
        loadingBar.visibility = View.GONE
        rootLayout.addView(loadingBar, paramsBar)

        val controlPanel = LinearLayout(this)
        controlPanel.orientation = LinearLayout.VERTICAL
        controlPanel.setBackgroundColor(Color.parseColor("#40000000"))
        controlPanel.setPadding(20, 20, 20, 20)
        val paramsPanel = FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        paramsPanel.gravity = Gravity.BOTTOM or Gravity.END
        paramsPanel.setMargins(0, 0, 50, 50)
        rootLayout.addView(controlPanel, paramsPanel)

        switchVideoBtn = Button(this)
        switchVideoBtn.text = "切换视频素材"
        switchVideoBtn.setOnClickListener { onSwitchVideoClicked() }
        controlPanel.addView(switchVideoBtn)

        colorFixBtn = Button(this)
        colorFixBtn.text = "颜色修正: 直通"
        colorFixBtn.setTextColor(Color.CYAN)
        colorFixBtn.setOnClickListener { onColorFixClicked() }
        controlPanel.addView(colorFixBtn)

        setContentView(rootLayout)
    }

    private fun copyRawResourceToCache(resName: String): String? {
        try {
            val resId = resources.getIdentifier(resName, "raw", packageName)
            if (resId == 0) return null
            val file = File(cacheDir, "$resName.mp4")
            if (file.exists() && file.length() > 0) return file.absolutePath
            val inputStream = resources.openRawResource(resId)
            val outputStream = FileOutputStream(file)
            inputStream.copyTo(outputStream)
            outputStream.close()
            inputStream.close()
            return file.absolutePath
        } catch (e: Exception) { return null }
    }

    private fun verifySdkStatus() { try { SDKManager.getInstance().sdkVersion } catch (e: Exception) {} }
    private fun appendStatus(text: String) { runOnUiThread { statusText.text = text } }
    override fun onDestroy() { super.onDestroy(); isRunning = false }
}