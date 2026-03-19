package dji.sampleV5.aircraft.graduation

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import org.opencv.core.Rect
import kotlin.math.min

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var procWidth: Int = 0
    private var procHeight: Int = 0

    private var targets: List<OverlayTarget> = emptyList()
    private var roiRect: Rect? = null
    private var modeText: String = "GLOBAL SEARCH"

    private val targetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val mainTargetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val roiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.FILL
        textSize = 30f
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    private val mainTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        textSize = 34f
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    private val modePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.FILL
        textSize = 38f
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    fun updateOverlay(
        procWidth: Int,
        procHeight: Int,
        targets: List<OverlayTarget>,
        roiRect: Rect?,
        modeText: String
    ) {
        this.procWidth = procWidth
        this.procHeight = procHeight
        this.targets = targets.map { it.deepCopy() }
        this.roiRect = roiRect?.let { Rect(it.x, it.y, it.width, it.height) }
        this.modeText = modeText
        postInvalidateOnAnimation()
    }

    fun clearOverlay() {
        targets = emptyList()
        roiRect = null
        modeText = "GLOBAL SEARCH"
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (procWidth <= 0 || procHeight <= 0 || width <= 0 || height <= 0) {
            return
        }

        val contentRect = computeContentRect(
            viewW = width.toFloat(),
            viewH = height.toFloat(),
            contentW = procWidth.toFloat(),
            contentH = procHeight.toFloat()
        )

        canvas.drawText(modeText, contentRect.left + 20f, contentRect.top + 40f, modePaint)

        roiRect?.let {
            canvas.drawRect(mapRect(it, contentRect), roiPaint)
        }

        val orderedTargets = targets.sortedBy { if (it.isMainTarget) 1 else 0 }
        orderedTargets.forEach { target ->
            val mapped = mapRect(target.rect, contentRect)
            val paint = if (target.isMainTarget) mainTargetPaint else targetPaint
            val textPaint = if (target.isMainTarget) mainTextPaint else this.textPaint

            canvas.drawRect(mapped, paint)

            val prefix = if (target.isMainTarget) "MAIN " else ""
            val text = "$prefix${target.label} ${"%.2f".format(target.confidence)}"
            val textX = mapped.left
            val textY = (mapped.top - 12f).coerceAtLeast(contentRect.top + 32f)
            canvas.drawText(text, textX, textY, textPaint)
        }
    }

    private fun computeContentRect(
        viewW: Float,
        viewH: Float,
        contentW: Float,
        contentH: Float
    ): RectF {
        val scale = min(viewW / contentW, viewH / contentH)
        val drawW = contentW * scale
        val drawH = contentH * scale
        val left = (viewW - drawW) / 2f
        val top = (viewH - drawH) / 2f
        return RectF(left, top, left + drawW, top + drawH)
    }

    private fun mapRect(src: Rect, contentRect: RectF): RectF {
        val scaleX = contentRect.width() / procWidth.toFloat()
        val scaleY = contentRect.height() / procHeight.toFloat()

        val left = contentRect.left + src.x * scaleX
        val top = contentRect.top + src.y * scaleY
        val right = contentRect.left + (src.x + src.width) * scaleX
        val bottom = contentRect.top + (src.y + src.height) * scaleY

        return RectF(left, top, right, bottom)
    }
}
