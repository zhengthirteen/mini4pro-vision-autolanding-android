package dji.sampleV5.aircraft.graduation

import org.opencv.core.Rect

data class OverlayTarget(
    val label: String,
    val confidence: Float,
    val rect: Rect,
    val isMainTarget: Boolean = false
) {
    val centerX: Double get() = rect.x + rect.width / 2.0
    val centerY: Double get() = rect.y + rect.height / 2.0

    fun deepCopy(isMainTarget: Boolean = this.isMainTarget): OverlayTarget {
        return copy(rect = Rect(rect.x, rect.y, rect.width, rect.height), isMainTarget = isMainTarget)
    }
}
