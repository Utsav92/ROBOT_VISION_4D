rv = op('/project1/ROBOT_VISION_4D')
out = rv.op('OUTPUT')
ow = out.op('output_window') or out.create(windowCOMP, 'output_window')
cw = rv.op('UI/control_window')
for w in (ow, cw):
    try:
        w.par.winclose.pulse()                          # close first so the new size/position is applied on reopen
    except Exception:
        pass
ow.par.winop = out.op('final_composite')
ow.par.winw, ow.par.winh = 930, 523                      # fits beside the 430 px control panel on a 1366x768 screen
ow.par.winoffsetx, ow.par.winoffsety = 0, 0
cw.par.winw, cw.par.winh = 430, 712
cw.par.winoffsetx, cw.par.winoffsety = 936, 0
ow.par.winopen.pulse()
cw.par.winopen.pulse()
rv.op('DATA_INPUT/timeline').module.reset()
rv.op('UI/panel').op('b_play').click()
result = 'output window 930x523 at (0,0); control panel 430x712 at (936,0); playing'
