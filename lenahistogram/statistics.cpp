#include<stdio.h>
#include<windows.h>//����bmp�ļ������ṹ
#include<graphics.h>//easyxͼ���
#include<conio.h>//Ϊ��ʹ��getch()����
#define scaley 10000 //��ͼʱ�Ŵ�����ݹ�ģ��




BITMAPFILEHEADER fileHeader;//bmp�ļ��ļ�ͷ����
BITMAPINFOHEADER infoHeader;//bmp�ļ���Ϣͷ����
RGBQUAD bmpColor[256];//bmp�ļ���ɫ�����
BYTE bmpValue[512*512];//�洢bmp�ļ�ͼ�����������

void showBmpHead(BITMAPFILEHEADER pBmpHead)//��ʾbmp�ļ�ͷ��������Ϣ
{

	printf("BMP�ļ���С��%dkb\n",fileHeader.bfSize/1024);
	printf("�����ֱ���Ϊ0��%d\n",fileHeader.bfReserved1);
	printf("�����ֱ���Ϊ0��%d\n",fileHeader.bfReserved2);
	printf("ʵ��λͼ���ݵ�ƫ���ֽ�����%d\n",fileHeader.bfOffBits);
}
void showBmpInfoHead(BITMAPINFOHEADER pBmpInfo)//��ʾbmp��Ϣͷ��������Ϣ
{
	printf("λͼ��Ϣͷ\n");
	printf("��Ϣͷ�Ĵ�С%db\n",infoHeader.biSize);
	printf("λͼ��ȷ�����������%d\n",infoHeader.biWidth);
	printf("λͼ�߶ȷ�����������%d\n",infoHeader.biHeight);
	printf("ʹ�õ���ɫ������%d\n",infoHeader.biClrUsed);
	printf("ͼ���С��%dkb\n",infoHeader.biSizeImage/1024);
	printf("ÿ����ռ��λ����%d\n",infoHeader.biBitCount);
}

int main()
{
	FILE* fp;//�����ļ�����
	int i;//ѭ����
	int column[256]={0};//�洢�����Ҷ���ͼ���г��ֵĴ���
	fp=fopen("lena.bmp","rb");
	fread(&fileHeader,1,sizeof(BITMAPFILEHEADER),fp);//��ȡ�ļ�ͷ
	showBmpHead(fileHeader);
	fread(&infoHeader,1,sizeof(BITMAPINFOHEADER),fp);//��ȡ��Ϣͷ
	showBmpInfoHead(infoHeader);
	fread(bmpColor,4,256,fp);//��ȡ��ɫ��
	fread(bmpValue,1,512*512,fp);//��ȡͼ�����ݲ���
	fclose(fp);//�ر��ļ�
	for(i=0;i<512*512;i++)
		column[bmpValue[i]]++;//ͳ�Ƹ����Ҷȳ��ֵĴ���
	for(i=0;i<256;i++)
		printf("%d:%d\n",i,column[i]);//�ı���ʽ��������Ҷȳ��ֵĴ���
	// ��������С��600�ķ�Χ��
	double frequency1[256];//�洢���ݵĳ���Ƶ�Σ���С��
	for(i=0;i<256;i++)
		frequency1[i]=column[i]/(256*256.0);
	int  frequency2[256];//�洢scaley��Χ�ڵ�����Ƶ�� Ϊ����������easyx�л�ͼʱ�������������
	for(i=0;i<256;i++)
		frequency2[i]=(int)(frequency1[i]*scaley);
	//��ֱ��ͼ
	void histogram(int*);//histogram��������
	histogram(frequency2);//����histogram����
	return 0;
}

void histogram(int* frequency)
{
	initgraph(800,800);//����ͼ�ν���
	line(100,700,100,0);//������
	line(100,700,800,700);//������
	int i;//ѭ����
	for(i=0;i<256;i++)
		rectangle(120+2*i,700-frequency[i],120+2*i+2,700);//ͨ�������������Ƴ�ֱ��ͼ
	getch();//��ֹͼ�ν����Զ���ʧ
}