package dji.sampleV5.aircraft.graduation

import android.util.Log
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import java.util.Arrays

class YoloV8Detector {
    private var net: Net? = null

    private val classNames = arrayOf(
        "blue_circle_2", "green_circle_3", "land_h",
        "purple_square", "red_circle_1", "yellow_triangle"
    )

    private val INPUT_SIZE = Size(640.0, 640.0)
    private val CONFIDENCE_THRESHOLD = 0.15f
    private val NMS_THRESHOLD = 0.45f

    fun init(modelPath: String): Boolean {
        return try {
            net = Dnn.readNetFromONNX(modelPath)
            Log.i("YoloV8", "模型加载成功: $modelPath")
            true
        } catch (e: Exception) {
            Log.e("YoloV8", "模型加载失败: ${e.message}")
            false
        }
    }

    fun detect(src: Mat): List<DetectionResult> {
        if (net == null || src.empty()) return emptyList()

        // 1. 预处理
        val blob = Dnn.blobFromImage(src, 1.0 / 255.0, INPUT_SIZE, Scalar(0.0, 0.0, 0.0), false, false)
        net!!.setInput(blob)

        // 2. 推理
        val outputBlob = net!!.forward() // 原始输出 [1, 10, 8400]

        // ===============================================================
        // 【关键修复 1】强制维度重塑 (Reshape)
        // OpenCV Java 处理 3D Mat 很麻烦，我们把它重塑为 2D 矩阵 (10行, 8400列)
        // 参数: cn=1 (保持单通道), rows=10 (你的属性数量)
        // 这样可以确保 get(0,0,data) 能正确把数据读出来
        // ===============================================================

        // 注意：这里假设你的输出是 [1, 10, 8400]。
        // 如果 outputBlob.dims() 是 3，我们需要小心处理。

        // 方案：直接展平为 1行 N列 的超长数组，然后自己算索引，这是最稳妥的
        val flatBlob = outputBlob.reshape(1, 1)
        val totalElements = flatBlob.total().toInt()
        val data = FloatArray(totalElements)
        flatBlob.get(0, 0, data)

        // ===============================================================
        // 【关键调试 2】打印原始数据采样 (看看是不是全是0)
        // ===============================================================
        // Log.d("YoloDebug", "Blob Shape: " + outputBlob.size().toString() + " Total: " + totalElements)
        // Log.d("YoloDebug", "Raw Data Sample (First 10): " + Arrays.toString(data.sliceArray(0..10)))
        // Log.d("YoloDebug", "Raw Data Sample (Middle): " + Arrays.toString(data.sliceArray(totalElements/2..totalElements/2+10)))

        // 3. 解析逻辑
        val numClasses = classNames.size
        val numAttributes = 4 + numClasses // 10
        val numAnchors = 8400

        // 检查数据量是否对齐
        if (totalElements != numAttributes * numAnchors) {
            Log.e("YoloDebug", "数据量不对齐！Expected: ${numAttributes * numAnchors}, Got: $totalElements")
            return emptyList()
        }

        val boxes = ArrayList<Rect2d>()
        val confidences = ArrayList<Float>()
        val classIds = ArrayList<Int>()

        var maxScoreGlobal = 0f

        // 遍历锚点
        // 数据布局: [1, 10, 8400] 展平后 -> 先排满第1行的8400个数据，再排第2行...
        // Row 0: cx values (0..8399)
        // Row 1: cy values (0..8399)
        // ...

        for (i in 0 until numAnchors) {
            var maxScore = 0f
            var maxClassId = -1

            // 遍历类别概率 (从第4行开始)
            for (c in 0 until numClasses) {
                val rowIdx = 4 + c
                // 计算在 1D 数组中的索引
                val index = rowIdx * numAnchors + i
                val score = data[index]

                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c
                }
            }

            if (maxScore > maxScoreGlobal) maxScoreGlobal = maxScore

            if (maxScore > CONFIDENCE_THRESHOLD) {
                // 提取坐标
                val cx = data[0 * numAnchors + i]
                val cy = data[1 * numAnchors + i]
                val w = data[2 * numAnchors + i]
                val h = data[3 * numAnchors + i]

                val x = (cx - w / 2) * (src.cols().toDouble() / INPUT_SIZE.width)
                val y = (cy - h / 2) * (src.rows().toDouble() / INPUT_SIZE.height)
                val width = w * (src.cols().toDouble() / INPUT_SIZE.width)
                val height = h * (src.rows().toDouble() / INPUT_SIZE.height)

                boxes.add(Rect2d(x, y, width, height))
                confidences.add(maxScore)
                classIds.add(maxClassId)
            }
        }

        // 打印每帧最高分，确认是否还在输出 0.0
        Log.d("YoloDebug", "MaxScore: $maxScoreGlobal | Boxes: ${boxes.size}")

        if (boxes.isEmpty()) {
            blob.release()
            outputBlob.release()
            flatBlob.release()
            return emptyList()
        }

        // NMS
        val boxesMat = MatOfRect2d()
        boxesMat.fromList(boxes)
        val confidencesMat = MatOfFloat()
        confidencesMat.fromList(confidences)
        val indices = MatOfInt()
        Dnn.NMSBoxes(boxesMat, confidencesMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices)

        val predictions = ArrayList<DetectionResult>()
        if (!indices.empty()) {
            val indicesArray = indices.toArray()
            for (idx in indicesArray) {
                val box = boxes[idx]
                val classId = classIds[idx]
                val score = confidences[idx]
                val rect = Rect(box.x.toInt(), box.y.toInt(), box.width.toInt(), box.height.toInt())

                predictions.add(DetectionResult(
                    classId = classId,
                    label = classNames[classId],
                    confidence = score,
                    rect = rect
                ))
            }
        }

        blob.release()
        outputBlob.release()
        flatBlob.release()
        boxesMat.release()
        confidencesMat.release()
        indices.release()

        return predictions
    }
}

// 确保这个类存在
data class DetectionResult(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val rect: Rect
)