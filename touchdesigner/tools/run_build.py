RV4D_DIR = r'C:\Users\Utsav\ROBOT_VISION_4D'
RV4D_SEQ = 0
if op('/project1') is None:
    root.create(baseCOMP, 'project1')
exec(open(RV4D_DIR + r'\touchdesigner\build_project.py', encoding='utf-8').read())
result = 'done'

