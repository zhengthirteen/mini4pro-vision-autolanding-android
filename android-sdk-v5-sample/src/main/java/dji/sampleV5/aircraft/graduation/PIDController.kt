package dji.sampleV5.aircraft.graduation

import kotlin.math.max
import kotlin.math.min

/**
 * 专为低频视觉设计的 PD 控制器 (综合优化版)
 * 1. 降低 Kp，增加 Kd 提供刹车阻尼
 * 2. 缩小死区修复状态机冲突 Bug
 * 3. 引入一阶低通滤波消除视觉检测框抖动带来的微分噪声
 */
class PIDController(
    private val maxSpeed: Double // 建议传入 0.15
) {
    // 调小 P，让动作更柔和；保留适量 D，增强对抗速度变化的阻力
    private val Kp = 0.07
    private val Kd = 0.10

    private var lastError: Double = 0.0
    private var lastTime: Long = 0
    private var lastOutput: Double = 0.0

    fun reset() {
        lastError = 0.0
        lastTime = 0
        lastOutput = 0.0
    }

    fun calculate(error: Double): Double {
        val currentTime = System.currentTimeMillis()

        // 1. 死区控制 (Deadband)
        // 【核心修复】：死区降为 0.04，必须小于状态机的 ALIGN_THRESHOLD (0.10)
        // 否则 PID 输出 0 但状态机仍未对准，会导致无人机靠惯性无控制漂移！
        if (kotlin.math.abs(error) < 0.05) {
            reset()
            return 0.0
        }

        // 2. 计算 P 项 (比例)
        val pOut = Kp * error

        // 3. 计算 D 项 (微分/阻尼)
        var dOut = 0.0
        if (lastTime > 0) {
            val deltaTime = (currentTime - lastTime) / 1000.0
            if (deltaTime > 0) {
                val derivative = (error - lastError) / deltaTime
                dOut = Kd * derivative
            }
        }

        lastError = error
        lastTime = currentTime

        val rawOut = pOut + dOut

        // 4. 一阶低通滤波 (Low-pass Filter)
        // 消除 YOLO 边界框在两帧之间微微跳动引发的突兀输出
        val alpha = 0.6
        val smoothOut = alpha * rawOut + (1 - alpha) * lastOutput
        lastOutput = smoothOut

        // 5. 限幅 (Clamping)
        return max(-maxSpeed, min(maxSpeed, smoothOut))
    }
}