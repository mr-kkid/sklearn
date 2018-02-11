#encoding=UTF-8
#数据预处理
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer#插入空值
from sklearn.preprocessing import LabelEncoder#顺序编码
from sklearn.preprocessing import OneHotEncoder#独热编码  用DataFrame创建数据  用独热编码  方便

csv_data='''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            0.0,11.0,12.0,
          '''
csv_data=unicode(csv_data)#python2.7中需要先编码
data=pd.read_csv(StringIO(csv_data))
print data
print '\n'
print data.values
print '\n'

csv_data=data.values

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer.fit(csv_data)

csv_data2=imputer.transform(csv_data)

print csv_data2
print '\n'

size_mapping={'XL':3,'L':2,'M':1}


df=pd.DataFrame([
                    ['green','M',10.1,'class1'],
                    ['red','L',13.5,'class2'],
                    ['blue','XL',15.3,'class1']
                ])

df.columns=['colors','size','price','classlabel']

print pd.get_dummies(df[['price','size','colors','classlabel']])#用DataFrame技术时，此方法只对字符串序列进行转换
print '-------------------------'+'\n'

print df
print '\n'

df['size']=df['size'].map(size_mapping)

print df
print '\n'

class_label=LabelEncoder()
df['classlabel']=class_label.fit_transform(df['classlabel'].values)
print df
print '\n'


X=df[['colors','size','price']].values
X[:,0]=class_label.fit_transform(X[:,0])

ohe=OneHotEncoder(categorical_features=[0])
print ohe.fit_transform(X).toarray()
print '\n'

print pd.get_dummies(df[['price','colors','size','classlabel']])#用DataFrame技术时，此方法只对字符串序列进行转换
