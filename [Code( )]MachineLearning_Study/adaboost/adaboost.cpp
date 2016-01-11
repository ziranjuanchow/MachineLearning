/*初学adaboost，希望能和大家交流 QQ：358807915
email：smile_spring@yeah.net
*/


#include <iostream>
#include <algorithm>
#include <functional>
#include<cmath>
#include <vector>
using std::vector ;
using namespace std;
#define FCOUNT 100//特征数
#define CCOUNT 30//弱分类器个数
#define PCOUNT 200//正样本数
#define NCOUNT 300//负样本数

struct sample
{
  int features[FCOUNT];//特征
	int pos_neg;//正0，负1
	float weight;//权值
	int result;//分类器的识别结果

};
struct weakClassifier
{
	int indexF;
	float threshold;
};
struct MySortFunction
{
	int m_n;
	MySortFunction(int n):m_n(n)
	{
	}
	bool operator()(sample&s1,sample&s2)const
	{
		return s1.features[m_n]<s2.features[m_n];
	}
};

//创建正样本
void CreatePos(vector<sample>&a)
{
	int i,j;
   for(i=0;i<PCOUNT;i++)
	{
		sample temp;
		temp.pos_neg=0;
		temp.weight=(float)1/(2*PCOUNT);
		temp.result =0;
		for(j=0;j<FCOUNT;j++)
			temp.features[j]=rand()%10;
		a.push_back(temp);
	}
}
float min(float a,float b)
{
	return(a<=b?a:b);
}
//创建负样本
void CreateNeg(vector<sample>&a)
{
	int i,j;
	for(i=0;i<NCOUNT;i++)
	{
		sample temp;
		temp.pos_neg=1;
		temp.weight=(float)1/(2*NCOUNT);
		temp.result =1;
		for(j=0;j<FCOUNT;j++)
			temp.features[j]=rand()%10;
		a.push_back(temp);
	}
}

//Training classifier
void Training(vector<sample>&a,vector<weakClassifier>&b,float*factors)
{
	int i,j;
	vector<sample>::size_type id=0,tcount=a.size();
	for(i=0;i<CCOUNT;i++)
	{
		weakClassifier temp;	
	    float totalWeights=0.0,totalPos=0.0,totalNeg=0.0,bPos=0,bNeg=0;//（当前样本之前的）正负样本权值和
		float e,thr,besterr=1.0;//训练单个分类器时用到的错误率，阈值，最小错误率
		float FThr[FCOUNT];//特征阈值
		float minErr=1.0;//所有特征的最小错误率
	    float beta;//更新权值所需系数

/*权重归一化*/
		for(id=0;id<tcount;id++)
		{
			totalWeights+=a[id].weight;
			if(a[id].pos_neg ) {
				totalNeg+=a[id].weight;
			} else {
				totalPos+=a[id].weight;
			}
				
		}//拿到样本中，总的权值之和，正样本的权值和，负样本的权值和。
		for(id=0;id<tcount;id++)
			a[id].weight /=totalWeights;	//权值归一化 

/*对每一特征训练一弱分类器*/
		//这个分类器就是用一个简单的阈值分类，现将样本进行排序，阈值的选取是在特征值上，而阈值选择的标准是错误判别的样本权重和。
		for(j=0;j<FCOUNT;j++)//上边是对总的样本进行操作，下边一部分，是对每个特征进行操作，对每个特征进行相应的分类。
		{
			//按特征j排序
			sort(a.begin (),a.end (),MySortFunction(j));//按照样本的某个特征进行排序
			besterr=1.0;
			//求单个弱分类器的阈值
			for(id=0;id<tcount;id++)//所有样本
			{
				if(a[id].pos_neg ) bNeg+=a[id].weight ;//当前负样本的权值之和
				else bPos+=a[id].weight ;//当前正样本的权值之和
				e=min((bPos+totalNeg-bNeg),(bNeg+totalPos-bPos));//算出取当前样本的特征值为阈值时，错误分类的概率是多少。
				if(id==0)
					thr=a[id].features [j]-0.5;
				else
				{
					if(id==tcount-1)
						thr=a[a.size()-1].features[j]+0.5;
					else
						thr=(a[id].features [j]+a[id-1].features [j])/2;
				}
				if(e<besterr)
				{
					besterr=e;
					FThr[j]=thr;	
					//cout<<FThr[j]<<" "<<j<<endl;
				}
			}
#if 0
			if (bNeg < 20) {
				cout << "ZSJ负样本："<< bNeg << endl;
			}
			
			// 测试用来打断点的。
			int breasdf = 21;
			++breasdf;
#endif

		}
		
/*选取最优分类器*/
		for(j=0;j<FCOUNT;j++)
		{
			float serror=0.0;				
			for(id=0;id<tcount;id++)
			{
				
				if(a[id].features [j]<=FThr[j]) a[id].result =0;//positive sample  根据第j个特征及其上边算出的阈值进行相应的分类，也就是说，在每次训练中，都要对每一个特征进行分类，然后选出特征最明显的某个特征作为判别的方案。
				else
					a[id].result =1;
				serror+=a[id].weight *abs(a[id].result -a[id].pos_neg );//错误分类总和。
			}
			if(serror<minErr)//选择错误分类最少的那个
			{
				minErr=serror;
				temp.indexF=j;
				temp.threshold=FThr[j];	
			}
			
		}	//这个步骤是利用权值来进行相应的弱分类器的训练。
		b.push_back (temp);//选出一个弱分类器 
		beta=minErr/(1-minErr);
		factors[i]=log(1/beta);

/*更新权值*/
		for(id=0;id<tcount;id++)
		{
			if(a[id].pos_neg ==a[id].result )
				a[id].weight *=beta;
		}		
	}//强分类器训练完毕
	
}
					


void main1()
{
	vector<sample>a;
	vector<weakClassifier>b;
	float factors[CCOUNT];//at
	CreatePos(a);//创建正样本
	CreateNeg(a);//创建负样本
	Training(a,b,factors);//训练分类器

	//查看训练出的分类器的参数
	int i=0;
	for(i=0;i<CCOUNT;i++)
	{
		cout<<"系数"<<factors[i]<<"特征"<<b[i].indexF <<"阈值"<<b[i].threshold<<endl;
	}
	
	getchar();
}