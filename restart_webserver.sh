#!/bin/bash

sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
pkill sudoku
pkill gunicorn

sudo systemctl daemon-reload
sudo systemctl start sudoku
sudo systemctl enable sudoku

sudo nginx -t
sudo systemctl restart nginx

echo "Waiting 3 seconds for gunicorn to restart"
sleep 3
sudo systemctl status sudoku

echo "Waiting 3 seconds for gunicorn and load logs"
sleep 3
sudo journalctl -u sudoku