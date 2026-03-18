package dji.sampleV5.aircraft.graduation

import org.opencv.calib3d.Calib3d
import org.opencv.core.*

/**
 * PnP 解算工具类 - 基于真实物理测量数据
 * 目标对象：直径 40cm 的橙色停机坪
 */
class PnPSolver {

    private val cameraMatrix: Mat
    private val distCoeffs: MatOfDouble
    private val objectPoints: MatOfPoint3f

    // 停机坪橙色圆盘直径 = 0.40 米
    // minAreaRect 会拟合圆的外接正方形，其边长等于圆的直径。
    // 半边长 = 0.40 / 2 = 0.20 米
    private val REAL_HALF_SIZE = 0.20

    init {
        // 图像分辨率: 640 x 360
        // fx = fy = (D_px * 1.0) / 0.40
        val fx = 440.0 //
        val fy = 440.0 //

        // 主点 (Principal Point): 图像中心
        // 对于 640x360 分辨率，中心为 (320, 180)
        // 这个假设对于 DJI 数字图传通常是足够精确的
        val cx = 320.0
        val cy = 180.0

        val cameraData = doubleArrayOf(
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        )
        cameraMatrix = Mat(3, 3, CvType.CV_64F)
        cameraMatrix.put(0, 0, *cameraData)

        // 2. 初始化畸变系数
        // DJI Mini 4 Pro 的 SDK 输出流通常已经过内部几何校正（去畸变）
        // 因此这里使用 0 畸变是符合工程实践的正确假设
        distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)

        // 3. 定义世界坐标系 (物理尺寸)
        // 以停机坪圆心为原点 (0,0,0)
        // 识别的是橙色圆盘(40cm)，拟合为正方形(40x40cm)
        // 顺序：左上 -> 右上 -> 右下 -> 左下
        val worldPoints = arrayOf(
            Point3(-REAL_HALF_SIZE, REAL_HALF_SIZE, 0.0),  // 左上 (-0.2, 0.2)
            Point3(REAL_HALF_SIZE, REAL_HALF_SIZE, 0.0),   // 右上 (0.2, 0.2)
            Point3(REAL_HALF_SIZE, -REAL_HALF_SIZE, 0.0),  // 右下 (0.2, -0.2)
            Point3(-REAL_HALF_SIZE, -REAL_HALF_SIZE, 0.0)  // 左下 (-0.2, -0.2)
        )
        objectPoints = MatOfPoint3f()
        objectPoints.fromList(worldPoints.toList())
    }

    /**
     * 执行 PnP 解算
     */
    fun solve(corners: Array<Point>): DoubleArray? {
        if (corners.size != 4) return null

        val imagePoints = MatOfPoint2f()
        imagePoints.fromArray(*corners)

        val rvec = Mat()
        val tvec = Mat()

        // 使用 SOLVEPNP_IPPE_SQUARE，这是针对平面4点目标最精确的算法
        val success = Calib3d.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec,
            false,
            Calib3d.SOLVEPNP_IPPE_SQUARE
        )

        var result: DoubleArray? = null

        if (success) {
            // tvec 输出单位与 objectPoints 一致 (米)
            val x = tvec.get(0, 0)[0]
            val y = tvec.get(1, 0)[0]
            val z = tvec.get(2, 0)[0]
            result = doubleArrayOf(x, y, z)
        }

        imagePoints.release()
        rvec.release()
        tvec.release()

        return result
    }

    fun release() {
        cameraMatrix.release()
        distCoeffs.release()
        objectPoints.release()
    }
}