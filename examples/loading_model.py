import sys, os
sys.path.insert(0, os.path.abspath('.'))
import mammopy as mi

model = mi.pre_process.load_model('base')

print(model)