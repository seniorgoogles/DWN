#!/bin/bash
rm -r dist && pip3 uninstall -y torch_dwn && python3 -m build --no-isolation  && pip3 install dist/*whl