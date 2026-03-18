package dji.sampleV5.aircraft.graduation

import kotlin.math.max
import kotlin.math.min

/**
 * 改进后的 PD 控制器
 * 增加了微分(D)项，用于抑制震荡
 */
class PIDController(
    private val maxSpeed: Double // 最大限速 (建议设为 0.5 或者更保守的 0.3)
) {
    // 参数调整建议：
    // Kp (拉力): 越大反应越快，但也容易震荡。建议 0.1 ~ 0.15
    // Kd (刹车): 用于抑制震荡。建议 0.05 ~ 0.1
    private val Kp = 0.12
    private val Kd = 0.08

    private var lastError: Double = 0.0
    private var lastTime: Long = 0

    // 重置状态 (每次开启追踪时建议调用，防止微分飞车)
    fun reset() {
        lastError = 0.0
        lastTime = 0
    }

    fun calculate(error: Double): Double {
        val currentTime = System.currentTimeMillis()

        // 1. 死区控制 (Deadband)
        // 误差小于 10cm 时忽略，防止在中心反复微调
        if (kotlin.math.abs(error) < 0.10) {
            reset()
            return 0.0
        }
        // 2. 计算 P 项 (比例)
        val pOut = Kp * error

        // 3. 计算 D 项 (微分/阻尼)
        var dOut = 0.0
        if (lastTime > 0) {
            val deltaTime = (currentTime - lastTime) / 1000.0 // 转换为秒
            if (deltaTime > 0) {
                // 微分公式: (当前误差 - 上次误差) / 时间间隔
                // 当误差快速减小时，(error - lastError) 为负，D项会产生反向力
                val derivative = (error - lastError) / deltaTime
                dOut = Kd * derivative
            }
        }

        // 更新状态
        lastError = error
        lastTime = currentTime

        // 4. 合成输出
        val totalOut = pOut + dOut

        // 5. 限幅 (Clamping)
        // 确保输出速度不超过设定的 maxSpeed
        return max(-maxSpeed, min(maxSpeed, totalOut))
    }
}