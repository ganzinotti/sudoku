#!/bin/bash

sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
sudo systemctl daemon-reload
sudo systemctl start sudoku
sudo systemctl enable sudoku
sudo nginx -t
sudo systemctl restart nginx
sudo journalctl -u sudoku