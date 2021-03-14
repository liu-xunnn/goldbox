#include<stdio.h>
#include<windows.h>//包含bmp文件各个结构
#include<graphics.h>//easyx图像库
#include<conio.h>//为了使用getch()函数
#define scaley 10000 //画图时放大的数据规模，




BITMAPFILEHEADER fileHeader;//bmp文件文件头对象
BITMAPINFOHEADER infoHeader;//bmp文件信息头对象
RGBQUAD bmpColor[256];//bmp文件颜色板对象
BYTE bmpValue[512*512];//存储bmp文件图像二进制数据

void showBmpHead(BITMAPFILEHEADER pBmpHead)//显示bmp文件头各部分信息
{

	printf("BMP文件大小：%dkb\n",fileHeader.bfSize/1024);
	printf("保留字必须为0：%d\n",fileHeader.bfReserved1);
	printf("保留字必须为0：%d\n",fileHeader.bfReserved2);
	printf("实际位图数据的偏移字节数：%d\n",fileHeader.bfOffBits);
}
void showBmpInfoHead(BITMAPINFOHEADER pBmpInfo)//显示bmp信息头各部分信息
{
	printf("位图信息头\n");
	printf("信息头的大小%db\n",infoHeader.biSize);
	printf("位图宽度方向像素数：%d\n",infoHeader.biWidth);
	printf("位图高度方向像素数：%d\n",infoHeader.biHeight);
	printf("使用的颜色数量：%d\n",infoHeader.biClrUsed);
	printf("图像大小：%dkb\n",infoHeader.biSizeImage/1024);
	printf("每像素占的位数：%d\n",infoHeader.biBitCount);
}

int main()
{
	FILE* fp;//定义文件对象
	int i;//循环数
	int column[256]={0};//存储各个灰度在图像中出现的次数
	fp=fopen("lena.bmp","rb");
	fread(&fileHeader,1,sizeof(BITMAPFILEHEADER),fp);//读取文件头
	showBmpHead(fileHeader);
	fread(&infoHeader,1,sizeof(BITMAPINFOHEADER),fp);//读取信息头
	showBmpInfoHead(infoHeader);
	fread(bmpColor,4,256,fp);//读取颜色板
	fread(bmpValue,1,512*512,fp);//读取图像数据部分
	fclose(fp);//关闭文件
	for(i=0;i<512*512;i++)
		column[bmpValue[i]]++;//统计各个灰度出现的次数
	for(i=0;i<256;i++)
		printf("%d:%d\n",i,column[i]);//文本形式输出各个灰度出现的次数
	// 将数据缩小到600的范围内
	double frequency1[256];//存储数据的出现频次，有小数
	for(i=0;i<256;i++)
		frequency1[i]=column[i]/(256*256.0);
	int  frequency2[256];//存储scaley范围内的数据频次 为整数，由于easyx中画图时坐标必须是整数
	for(i=0;i<256;i++)
		frequency2[i]=(int)(frequency1[i]*scaley);
	//画直方图
	void histogram(int*);//histogram函数声明
	histogram(frequency2);//调用histogram函数
	return 0;
}

void histogram(int* frequency)
{
	initgraph(800,800);//创建图形界面
	line(100,700,100,0);//纵坐标
	line(100,700,800,700);//横坐标
	int i;//循环数
	for(i=0;i<256;i++)
		rectangle(120+2*i,700-frequency[i],120+2*i+2,700);//通过画矩形来绘制出直方图
	getch();//防止图形界面自动消失
}