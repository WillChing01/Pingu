del /Q /S %cd%\_raw && ^
python download.py -d _raw && ^
del /Q /S %cd%\dataset\training && ^
del /Q /S %cd%\dataset\validation && ^
make clean && ^
make && ^
parse.exe -N 6
