import sensor, lcd
import KPU as kpu
import utime as time
from machine import Timer

from network_esp32 import enable_esp32
import http


enable_esp32()

# base_url = "http://35.192.65.36:80"
base_url = "http://192.168.0.181:8000"

exp = "edge"  # cloud or edge


exp_id = http.post(base_url + "/experiment/new")
exp_id = exp_id.json()["id"]
print(exp_id)

with open("local-%s_%s.csv" % (exp, exp_id), "w") as f:
    f.write("start,sensor,pred,req,total\n")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(False)
sensor.set_vflip(False)
sensor.run(1)

lcd.init(type=1)

lcd.clear(lcd.WHITE)



anchors = (
    1.889,
    2.5245,
    2.9465,
    3.94056,
    3.99987,
    5.3658,
    5.155437,
    6.92275,
    6.718375,
    9.01025,
)

task = kpu.load(0x300000)
kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)




def on_timer(timer):
    t = time.ticks_ms()
    img = sensor.snapshot()
    s_t = time.ticks_ms()
    objects = kpu.run_yolo2(task, img)
    p_t = time.ticks_ms()
    body = {"result": objects if objects else [], "time": t, "net": "yolov2"}
    req = http.post(base_url + "/result/save", json=body)
    r_t = time.ticks_ms()
    if objects:
        for obj in objects:
            img.draw_rectangle(obj.rect())
    img.draw_string(0, 200, "t:%dms" % (time.ticks_ms() - t), scale=2, color=(255, 0, 0))
    lcd.display(img)
    with open("local-%s_%s.csv" % (exp, exp_id), "a") as f:
        f.write("%d,%d,%d,%d,%d\n" % (t, s_t, p_t, r_t, time.ticks_ms()))


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
