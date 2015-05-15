mount /dev/sda /media/usb
echo "Finished mounting"
cp /media/usb/MENG_Project/kernel .
cp /media/usb/MENG_Project/kernel.aocx .

aocl program /dev/acl0 kernel.aocx

echo "Copying over 1600"
cp /media/usb/MENG_Project/L_1600.BMP left.BMP
cp /media/usb/MENG_Project/R_1600.BMP right.BMP

./kernel

echo "Copying over 1280"
cp /media/usb/MENG_Project/L_1280.BMP left.BMP
cp /media/usb/MENG_Project/R_1280.BMP right.BMP

./kernel

echo "Copying over 1024"
cp /media/usb/MENG_Project/L_1024.BMP left.BMP
cp /media/usb/MENG_Project/R_1024.BMP right.BMP

./kernel

echo "Copying over 720"
cp /media/usb/MENG_Project/L_720.BMP left.BMP
cp /media/usb/MENG_Project/R_720.BMP right.BMP

./kernel

echo "Copying over 640"
cp /media/usb/MENG_Project/L_640.BMP left.BMP
cp /media/usb/MENG_Project/R_640.BMP right.BMP


./kernel

cp FPGA_Bilateral_Filter_Left.bmp /media/usb/MENG_Project
cp FPGA_Bilateral_Filter_Right.bmp /media/usb/MENG_Project
cp FPGA_Depth.bmp /media/usb/MENG_Project

umount /media/usb

