@echo off
:: ���Ҫ�ļ�������·��������Ҫ��dir��һ���%%~nxi�����Ķ�
::                  code by FBY && RMW
if exist FileNameList.txt del FileNameList.txt /q
::for /f "delims=" %%i in ('dir *.jpg /b /a-d /s') do echo %%~nxi >>FileNameList.txt
for /f "delims=" %%i in ('dir *.jpg /b /a-d /s') do (
			echo %%~dpi%%~nxi >>FileNameList.txt
			echo 2 >>FileNameList.txt
			)
if not exist FileNameList.txt goto no_file
start FileNameList.txt
exit

:no_file
cls
echo       %cur_dir% �ļ�����û�е������ļ�
pause 