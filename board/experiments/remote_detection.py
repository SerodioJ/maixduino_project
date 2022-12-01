import sensor, lcd
import utime as time
from machine import Timer
from ubinascii import b2a_base64

from network_esp32 import enable_esp32
import http

enable_esp32()


# base_url = "http://35.192.65.36:80"
base_url = "http://192.168.0.181:8000"

exp = "cloud"  # cloud or edge

exp_id = http.post(base_url + "/experiment/new")
exp_id = exp_id.json()["id"]

with open("%s_%s.csv" % (exp, exp_id), "w") as f:
    f.write("start,sensor,req,total\n")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.run(1)

lcd.init(type=1)
lcd.clear(lcd.WHITE)


def on_timer(timer):
    t = time.ticks_ms()
    img = sensor.snapshot()
    body = {
        "image": b2a_base64(img.compressed().to_bytes()),
        "time": t,
    }
    s_t = time.ticks_ms()
    req = http.post(base_url + "/predict/detection", json=body)
    r_t = time.ticks_ms()
    objects = req.json()
    img.draw_string(
        0, 200, "t:%dms" % (time.ticks_ms() - t), scale=2, color=(255, 0, 0)
    )
    lcd.display(img)
    with open("%s_%s.csv" % (exp, exp_id), "a") as f:
        f.write("%d,%d,%d,%d\n" % (t, s_t, r_t, time.ticks_ms()))


tim = Timer(
    Timer.TIMER0,
    Timer.CHANNEL0,
    mode=Timer.MODE_PERIODIC,
    period=10,
    unit=Timer.UNIT_S,
    callback=on_timer,
    arg=None,
    start=True,
    priority=1,
    div=0,
)
