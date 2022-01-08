import os

os.system("rmdir /s /q build")
os.system("rmdir /s /q dist")
os.system("python setup.py bdist_wheel")

whl = os.listdir("./dist")[0]
path = os.path.join(os.getcwd(), "dist", whl)

os.system("pip uninstall -y datasets")
os.system("pip install %s" % path)
