package dji.sampleV5.aircraft.graduation

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import dji.sampleV5.aircraft.R

class LauncherMenuActivity : AppCompatActivity() {

    private lateinit var btnSim: Button
    private lateinit var btnReal: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_launcher_menu)

        btnSim = findViewById(R.id.btn_open_simulation)
        btnReal = findViewById(R.id.btn_open_realtime)

        btnSim.setOnClickListener {
            startActivity(Intent(this, VisionSimulationActivity::class.java))
        }

        btnReal.setOnClickListener {
            startActivity(Intent(this, RealTimeTrackingActivity::class.java))
        }
    }
}