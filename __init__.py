import os, sys, inspect


sys.path.append(os.path.dirname(os.path.realpath(__file__)))



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)