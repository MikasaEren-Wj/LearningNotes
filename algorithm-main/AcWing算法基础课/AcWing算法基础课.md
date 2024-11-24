万能头文件 `#include<bits/stdc++.h>`包含所有C++的库函数

**使用`ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);`加快C++中cin、cout的效率，此时不能使用scanf和printf**

# 第一章 基础算法

## 1.排序

### 快速排序（AcWing785）

**主要思想**：基于分治思想

`l`是待排序区间的左边界，`r`是右边界

1. 确定分界点`x`，可以取左边界的值`q[l]`，或右边界的值`q[r]`，或者中间位置的值`q[(l + r)/2]`

2. 根据基准值，调整区间，使得左半边区间的值全都`≤ x`，右半边区间的值全都`≥ x` 

   采用双指针：左指针`i`从左边界`l-1`开始，往右扫描，右指针`j`从右边界`r+1`开始，往左扫描 

   > 为什么初始是l-1，r+1？
   >
   > 因为后续的不管执行交换与否，首先都先将i，j向内移动一位，所以一开始的两个指针都设置为超出边界一个位置

   当满足条件`q[i] < x`时，`i`右移；直到不满足条件时，即`q[i] >= x`，`i`停下;

   然后移动右指针`j`，`j` 当满足条件`q[j] > x`时，`j`左移；直到不满足条件时，即`q[j] <= x`，`j`停下；

   交换`q[i]`和`q[j]` 将`i`右移一位，`j`左移一位，重复上面的操作，直到`i`和`j`相遇`(最终i和j的位置为：i==j或i=j+1)`。 此时左半区间的数都满足`≤x`，且左半区间的最后一个数的下标为`j`，右半区间的数都满足`≥ x`,且右半区间的第一个数的下标为`i`

3. 递归处理左右两段，

   若用`j`来作为区间的分界，则`[l, j]` 都是`≤x`，`[j + 1, r]`都是`≥x`

   若用`i`来作为区间的分界，则`[l, i - 1]`都是`≤x`，`[i, r]`都是`≥x`

> 注意：
>
> 递归取`[l, j]`，`[j + 1, r]`区间时，基准值不能取右边界`x=q[r]`，不然会出现死循环问题，此时常取左边界x=q[l]或中间值  （eg:1,2 会出现死循环）
>
> 同理当递归取`[l, i - 1]`，`[i,r]`区间时，基准值不能取左边界`x=q[l]`，不然会出现死循环问题，此时常取左边界x=q[r] 或中间值  （eg:1,2 会出现死循环）
>
> 快排不稳定，平均时间复杂度nlogn

```c++
#include <iostream>
#include <algorithm> //包含swap函数
using namespace std;
const int N=1e6+10;
int n;
int a[N];

void quick_sort(int a[],int l,int r){
    if(l>r) return;
    
    int x=a[(l+r)/2],i=l-1,j=r+1;
    while(i<j){
        while(a[++i]<x);
        while(a[--j]>x);
        if(i<j) swap(a[i],a[j]);
    }
    //递归
    quick_sort(a,l,j); //或[l,i-1]
    quick_sort(a,j+1,r); //或[i,r]
}



```

#### **AcWing.786 求第k个数**

> 第k个数
>
> 给定一个长度为 n 的整数数列，以及一个整数 k，请用快速选择算法求出数列从小到大排序后的第 k 个数。
>
> 输入格式
> 第一行包含两个整数 n 和 k。
>
> 第二行包含 n 个整数（所有整数均在 1∼109 范围内），表示整数数列。
>
> 输出格式
> 输出一个整数，表示数列的第 k 小数。
>
> 数据范围
> 1≤n≤100000,
> 1≤k≤n
>
> 输入样例：
> 5 3
> 2 4 1 5 3
>
> 输出样例：
> 3

**实现思路：**使用基于快速排序的快速选择思想(O(n))，相比直接用快速排序输出第k个数(O(nlogn))时间复杂度更小

- 根据快速排序的思想，将数组划分为两个区间，分别递归两个区间，直到左区间<=基准值，右区间>=基准值。
- 设左区间的数据个数为`sl`，右区间的数据个数为`sr`。
- 则若`k<=sl`，就意味着最终答案在左区间，则后续无需再递归右区间排序，**只需递归左区间**，输出左区间的第`k`个数即为结果
- 同理，若`k>sl`，就意味着最终答案在右区间，则后续无需再遍历左区间，**只需递归右区间**，输出右区间第`k-sl`个数即为结果。

```c++
#include <iostream>
using namespace std;
int N=1e6+10;
int n,k;
int a[N];

//返回最终结果
int quick_sort(int l,int r,int k){
	//先正常进行快排
    if(l==r) return a[l];//若区间中只有一个数 直接返回为结果
    
    int i=l-1,j=r+1,x=a[(l+r)/2];
    while(i<j){
        while(a[++i]<x);
        while(a[--j]>x);
        if(i<j) swap(a[i],a[j]);
    }
    if(k<=j) return quick_sort(l,j,k);//结果在左区间，只递归左区间
    return quick_sort(j+1,r,k-sl);//结果在右区间，只递归右区间
}

int mian(){
    cin>>n>>k;
    for(int i=0;i<n;i++) cin>>a[i];
    cout<<quick_sort(0,n-1,k)<<endl;
    return 0;
}
```



### 归并排序（AcWing787）

**主要思想**：也是基于分治思想

1.先确认分界点，一般取中间点`(l+r)/2`

2.对左右两个区间分别递归排序

3.将两侧排好序的两个有序数组合二为一

实现思路：设置左右指针和一个临时数组temp，左指针指向左区间的的第一个元素，右指针指向右区间的第一个元素，循环比较左右指针所指元素，两者较小的元素放入temp数组中，指针后移继续比较。直至某一指针到达末尾，将其中一个未放置完的区间的数再都放入temp数组。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1e6+10;
int n;
int a[N]，temp[N];

void merge_sort(int a[],int l,int r){
	if(l>r) return;
    
    int mid=(l+r)/2;
    merge_sort(a,l,mid),merge_sort(a,mid+1,r);//递归左右区间
    int k=0,i=l,j=mid+1;//k为临时数组的指针，i为左区间指针，j为右区间指针
    while(i<=mid && j<=r){
        if(a[i]<=a[j]) temp[k++]=a[i++];
        else temp[k++]=a[j++];
    }
    while(i<=mid) temp[k++]=a[i++];//左区间有剩余
    while(j<=r) temp[k++]=a[j++];//右区间有剩余
    
    //最后将temp数组的数据放回a数组
    for(int i=l,k=0;i<=r;i++,k++)
        a[i]=temp[k];
    
}


```

> 注意：
>
> 归并排序稳定，时间复杂度为nlogn

#### AcWing.788 求逆序对数量

![image-20240806112010719](assets\image-20240806112010719.png)

实现思路：

- 根据**归并排序**的思想，将数组分为各自有序的左右两个区间`[l,mid]`,`[mid+1,r]`，采用双指针开始分别指向两个区间的第一个元素，相互比较选出较小的那个元素，然后后移，不断循环，直到一个区间遍历完。
- 在比较过程中，设`i`指向左区间，`j`指向右区间，由于两个区间各自有序，逆序对只会出现一种情况，即左区间存在大于右区间元素的元素。
- 若`a[i]>a[j]`，则左区间中从`i`开始到`mid`的元素都大于`a[j]`，与`a[j]`组成逆序对，数量为`mid-i+1`

> **注意：**对于给定n个数，最坏的情况为逆序，则逆序对数为n(n-1)/2个，题中数据个数范围为100000，则最大结果会超出int的存储范围(-2^31~2^31-1)，所以虽好使用long long来存储最终结果

```c++
#include <iostream>
using namespace std;
typedef long long LL;

int N=100010;
int a[N],tmp[N];
int n;

LL merge_sort(int l,int r){
    if(l>=r) return 0;
    
    int mid=l+r>>1;
    LL res=merge_sort(l,mid)+merge_sort(mid+1,r);//左右区间分别进行归并
    
    //归并过程
    int k=0,i=l,j=mid+1;
    while(i<=mid && j<=r){
        if(a[i]<=a[j]) tmp[k++]=a[i++];
        else{
            tmp[k++]=a[j++];
            res+=mid-i+1;
        }
    }
    
    //某个区间剩余元素
    while(i<=mid) tmp[k++]=a[i++];
    while(j<=r) tmp[k++]=a[j++];
    
    //物归原主
    for(int i=l,j=0;i<n;i++,j++) a[i]=tmp[j];
    
    return res;
}

int mian(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
    cout<<merge_sort(0,n-1)<<endl;
    return 0;
}

```





## 2. 二分查找

### 整数二分

二分的本质不是**单调性**，有单调性一定可以二分，可以二分不一定有单调性 二分的本质是**边界**，假设给定一个区间，如果能够根据某个条件，将区间划分为左右两部分，使得左半边满足这个条件，右半边不满足这个条件（或者反之）。就可以用二分来查找**左右两部分的边界点**。

![img](assets\68747470733a2f2f706963342e7a68696d672e636f6d2f38302f76322d31623061623564656234396561343162393838333563396165313239373732642e706e67)

**主要思想：假设这组数据关键字已有序**

**1.寻找红色区间的右边界点   模板一**

> 当`l<r`时不断循环
>
> 先取中间值`mid=(l+r)/2`
>
> 先判断mid是否满足条件（即某种性质），`check(mid)`
>
> - **若满足**，`check(mid)`为true，区间更新为`[l,mid]`，即`r=mid`；
> - 若不满足，`check(mid)`为false，更新区间为`[mid+1,r] `，即`l=mid+1`。（注意这里`mid+1`，是因为mid本身已不满足条件，只能右移一个位置去区间找答案。）
>
> 最后`l=r`，循环结束，输出分界点位置`l`(或`r`)

**代码模板：**

```c++

int binarysearch_1(int l,int r){
    while(l<r){
        int mid=(l+r)/2;
        if(check(mid)) r=mid;
        else l=mid+1;
    }
    return l;//return r;也可
}
```

**2.寻找绿色区间的左边界点   模板二**

> 当l<r时不断循环
>
> 先取中间值`mid=(l+r+1)/2`
>
> - 注意，当采用`l = mid`和`r = mid - 1`这种更新方式时，计算`mid`时，要加上`1`（向上取整），即`mid = l + r + 1 / 2`。否则，在`l = r - 1`时，计算`mid`时若不加`1`，则`mid = l + r / 2 = l`，这样更新`l = mid`，就是`l = l`，会导致死循环。所以要向上取整，采用`mid = l + r + 1 / 2`。
>
> 先判断mid是否满足条件（即某种性质），`check(mid)`
>
> - **若满足**，`check(mid)`为true，区间更新为`[mid,r]`，即l=mid；
> - 若不满足，`check(mid)`为false，更新区间为`[l,mid-1] `，即r=mid-1。（注意这里mid-1，是因为mid本身已不满足条件，只能左移一个位置去区间找答案。）
>
> 最后`l=r`，循环结束，输出分界点位置l(或r)

**代码模板：**

```c++
int binarysearch_2(int l,int r){
    while(l<r){
        int mid=(l+r+1)/2;//需要加1，避免边界问题
        if(check(mid)) l=mid;
        else r=mid-1;
    }
    return l;
}
```



#### （AcWing 789.数的范围）

![img](assets\5bc7bf351a974840a45d492e1dc24ddb.png)

根据上述思想和模板，代码思路为：

对于模板一，check条件为`x<=a[mid]`，`mid=(l+r)/2`，满足条件时x在mid左侧，更新`r=mid`，最终输出为起始位置

对于模板二，check条件为`x>=a[mid]`，`mid=(l+r+1)/2`，满足条件时x在mid右侧，更新`l=mid`，最终输出为终止位置

最终依次输出各模板的`l`(或r)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
int n;//数组个数
int a[N];
int m,x;//m为询问个数，x为询问数
int main(){
    cin>>n>>m;
    for(int i=0;i<n;i++) cin>>a[i];
    while(m--){
        cin>>x;
        int l=0,r=n-1;
        while(l<r){
            int mid=(l+r)/2;
            if(a[mid]>=x) r=mid;
            else l=mid+1;
        }
        if(a[l]!=x) cout<<"-1 -1"<<endl;//未找到
        else{
            cout<<l<<" ";//起始位置
            int l=0,r=n-1;
            while(l<r){
                int mid=(l+r+1)/2;
                if(a[mid]<=x) l=mid;
                else r=mid-1;
            }
            cout<<l<<endl;//终止位置
        }
    }
    return 0;
}


```



### 浮点数二分

主要思想：相比整数二分，浮点数二分无需考虑边界问题，比较简单。当二分的区间足够小时，可以认为已经找到了答案，如当r - l < 1e-6（实际上就是区间长度比较小就近似认为是一个数） ，停止二分；或者直接迭代一定的次数，比如循环100次后停止二分

**如问题：给定一个浮点数n，求他的二次方根（三次方根类似 ）**

代码思路：使用整数二分中的哪个模板都可。这里使用模板一，check条件为x<=a[mid]，mid=(l+r)/2，满足条件更新r=mid

**注意**：r不能取小于1的数（小于1的数开方反而更大）.若x=0.04，取r=x，实际结果应为0.2，但算法会在0~0.04中找答案，无法找出答案，故应令r=max(1,x);

```c++
#include <iostream>
#include <algorithm>
using namespace std;
double x;
int main(){
    cin>>x;
    double l=0,r=max(1,x);
    while(l<r){
        int mid=(l+r)/2;
        if(mid*mid >=x) r=mid;
        else l=mid+1;
    }
    cout<<l;
    return 0;
}

```



## 3.高精度

主要有四种情况：

- A + B：两个大整数相加
- A - B：两个大整数相减
- A × b：一个大整数乘一个小整数
- A ÷ b：一个大整数除以一个小整数

**大整数的存储（加减乘除）**：C++和C中不支持直接存储大整数，需要自己定义用数组存储，数字的低位存储在数组的前面，高位存储在数组的后面，即倒序存入，对于整数“123456789”，数组a[0]存储“9”，a[8]存储“1”，以此类推，**方便进位**。（**即小端存储**）

**注意**：**a[i]为字符串，要转化为整数，需要执行`a[i]-'0'`**

代码实现需要用到C++中的**vector库**

> `vector` 是 C++ 标准库中的一个动态数组容器，位于 `<vector>` 头文件中。它提供了灵活且高效的方式来存储和管理一组元素。`std::vector` 可以动态调整大小，并提供了方便的元素访问、插入、删除和其他操作。
>
> **`push_back(const T& value)`**: 在 vector 末尾添加一个元素。
>
> **`pop_back()`**: 移除 vector 末尾的元素。
>
> **`size()`**: 返回 vector 中的元素个数。

### 1.高精度加法（A+B)

**主要思想**：

A、B两个整数分别存于`a[]`，`b[]`。设置一个结数组`c[]`，设置一个进位`t`，表示两个整数对应位相加时的进位，初始为0，后续每次相加结果为`a[i]+b[i]+t`。相加结果`%10`为余数，存入`c[i]`；相加结果`/10`为进位值即`t`。

**大整数加大整数(AcWing 791.高精度加法)**

![img](assets\2e3fbfa4dbeb41acbd4401a2805a2e26.png)

```c++
#include <iostream>
#include <vector>
using namespace std;
string a,b;
vector<int> A,B;

vector<int> add(vector<int> &A,vector<int> &B){ //函数传入参数时使用引用通常会比传值更快，尤其是在传递较大的对象时。这是因为引用避免了对象的拷贝操作，而拷贝大对象可能会耗费大量时间和资源。
    vector<int> C;
    int t=0;//进位标志
    for(int i=0;i<A.size()||i<B.size();i++){
        if(i<A.size()) t+=A[i];
        if(i<B.size()) t+=B[i];
        C.push_back(t%10);
        t/=10;
    }
    if(t) C.push_back(t);//最后还有进位
    return C;
}


int main(){
    cin>>a>>b;
    //逆置字符串存储到数组中
    for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');//a[i]-'0'转化为整数存入数组中
    for(int i=b.size()-1;i>=0;i--) B.push_back(b[i]-'0');
    vector<int> C;
    c=add(A,B);
    for(int i=C.size()-1;i>=0;i--) cout<<C[i]<<endl;
    return 0;
}

```



### 2.高精度减法（A-B)

**主要思想：**

1.首先要判断A和B的大小，A>B正常相减；B>A时，交换A和B的传入函数的顺序，并在输出时添加负号。

- 判断A和B的大小，首先比较A和B的位数，若A的位数多则A大
- 若位数相同，从高位开始逐位比较二者对应位置的数大小

2.实现相减：类似加法，设置一个结果数组`c[]`，设置一个借位标志`t`，初始为0。

- 每次减法为`a[i]-b[i]-t`（注意要判断B当前位置是否有数），结果赋予`t`。
- 再判断`t`大于或小于0。若`t`小于0，则`t+10`（即借位）；若`t`大于0，则无需借位值为`t`，对应结果存于`c[i]`。
- 最后判断`t`若小于0，将`t`赋值为1，表示低位向高位借位。

> 注意：
>
> 判断t大于小于0得到结果的操作可以简化为`(t+10)%10`
>
> 最后对于结果数组c[]要去除前导0

**大整数减大整数(AcWing 792.高精度减法)**

![img](assets\1deadaebbbbd4143a129b33f1c3099da.png)

```c++
#include <iostream>
#include <vector>
using namespace std;
vector<int> A,B;
string a,b;

//比较A、B大小
bool compare(vector<int> &A,vector<int> &B){
    if(A.size()!=B.size()) return A.size()>B.size();
    else{
		for(int i=A.size()-1;i>=0;i--)
            if(A[i]!=B[i]) return A[i]>B[i];
    }
    return true;//相等
}

vector sub(vector<int> &A,vector<int> &B){
	vector<int> C;
    int t=0;
    for(int i=0;i<A.size();i++){
        t=A[i]-t;
        if(i<B.size()) t-=B[i];
        C.push_back((t+10)%10);//这里直接一步到位处理t>=0和t<0的情况
        if(t>=0) t=0;//无需借位
        else t=1;
    }
    //去除前导零
    while(C.size()>1&&C.back()==0) C.pop_back();
    return C;
}

int main(){
    cin>>a>>b;
    for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
    for(int i=b.size()-1;i>=0;i--) B.push_back(b[i]-'0');
    vector<int> C;
    if(compare(A,B)){
        C=sub(A,B);
    }else{
        C=sub(B,A);
        cout<<"-";
    }
    for(int i=C.size()-1;i>=0;i--) cout<<C[i];
}


```

### 3.高精度乘法(A*b)

**主要思想：**

1.一个大整数乘以较小的整数。大整数依旧用数组`a[]`表示，较小的整数用一个整型变量表示b。

2.用大整数的每一位`a[i]`去乘以一个整数b。依旧设置一个结果数组`c[]`，设置一个进位`t`，初始为0。

- 每次计算`a[i]*b+t`，结果赋予`t`。`t%10`为对应位的结果，存入`c[i]`；`t=t/10`为进位值。（代码其实类似高精度加法）

![img](assets\a0e4f0d3fdd04e58b1365b0c8b381632.png)

```c++
#include <iostream>
#include <vector>
using namespace std;
string a;
int b;
vector<int> A;

vector mul(vector<int> &A,int b){
    int t=0;
    vector<int> C;
    for(int i=0;i<A.size() || t;i++){ //循环中判断t若大于0，意味着还有进位，为了处理最后对应数都乘完，但还有进位的情况
        if(i<A.size())t+=A[i]*b; //因为循环中判断中加入了t，这里需要重新判断一下条件
        C.push_back(t%10);
        t/=10;
    }
    return C;
}

int main(){
    cin>>a>>b;
    for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
    vector C=mul(A,b);
    for(int i=C.size()-1;i>=0;i--) cout<<C[i];
    return 0;
    
}

```



### 4.高精度除法（A÷b)

**主要思想：**

1.一个大整数除以较小的整数。大整数依旧用数组`a[]`表示，较小的整数用一个整型变量表示b。

2.除法的计算和加减乘有所不同，除法是从高位开始计算，因此循环要从数组`a[]`的末尾开始操作。设置存储商的结果数组`c[]`，设置存储余数的整数`r`，初始值为0。

- 上一位的余数r*10+当前位的数据a[i]，结果除以除数b即为当前位置的商a[i]
- 上一位的余数r*10+当前位的数据a[i]，结果对除数b取余即为当前位置的余数r

> 注意：商可能存在0，最后记得去掉结果数组c[i]中的前导零

```c++
#include <iostream>
#include <vector>
#include <algorithm> //包含函数reverse
using namespace std;
string a;
int b,r;//r为结果余数
vector<int> A;

vector div(vector<int> &A,int b,int &r){
    r=0;
    vector<int> C;
    for(int i=A.size()-1;i>=0;i--){//从数组末尾开始，即整数高位
        r=r*10+A[i];
        C.push_back(r/b);
        r%=b;
    }
    reverse(C.begin(),C.end());//因为C从高位存储结果，所以需逆置
    //去除前导零
    while(C.size()>1&&C.back()==0) C.pop_back();
    return C;
}

int mian(){
    cin>>a>>b;
    for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
    vector<int> C=div(A,b,r);
    for(int i=C.size()-1;i>=0;i--) cout<<C[i]<<endl;
    cout<<r;
    return 0;
}
```



## 4.前缀和与差分

### 1.一维前缀和

**一维前缀和**：S[i]=a1+a2+a3+a4+.....+ai，要求a从a1开始，且S[0]=0

> **前缀和的作用**：给定一组序列数据，可以计算任意第l个数到第r个数的和，S[r]-S[l-1]（这里就解释了为什么要求S[0]=0，因为当l=1时，实质就是求是S[r]）
>
> 求很多个任意子序列的数据和时，假如不使用前缀和公式，就需要顺序遍历来求，时间复杂度为O(n)
>
> 使用前缀和，直接得到结果，时间复杂度为O(1)

**AcWing 795.前缀和**

![img](assets\d8bf64afb6ef40fcbdd82983e8328812.png)

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int n,m;
int a[N],s[N];
int main(){
    cin>>n>>m;
    //注意数组下标从1开始存储
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    for(int i=1;i<=n;i++) S[i]=S[i-1]+a[i];//因为S[]为全局数组，元素默认初始化为0
    while(m--){
       int l,r;
        cin>>l>>r;
        cout<<S[r]-S[l-1]<<endl;
    }
    return 0;
}

```

### 2.二维前缀和

![image-20240714201113625](assets\image-20240714201113625.png)

**二维前缀和：**`S[i][j]`即为坐标`(i,j)`左上部分所有元素的和，也就是绿色区域的所有元素和（注意`i`，`j`依旧是从1开始，`s[0][0]`默认为0）

![image-20240714201618960](assets\image-20240714201618960.png)

**二维前缀和计算公式：**`s[i][j]`=`s[i][j-1]+s[i-1][j]-s[i-1][j-1]+a[i][j]`（注意这里要`s[i][j-1]+s[i-1][j]`多算了一次`s[i-1][j-1]`，所以要减去一次）

**任意子矩阵中所有数的和计算：**

设所求子矩阵的左上角坐标为`(x1,y1)`，右下角坐标为`(x2,y2)`

`s=s[x2][y2]-s[x1-1][y2]-s[x2][y1-1]+s[x1-1][y1-1]`

eg:`s=S[5][3] - S[2][3] - S[5][1] + S[2][1]`

![image-20240714202122530](assets\image-20240714202122530.png)

**AcWing 796.子矩阵的和**

![img](assets\9b1878af4f184e2a8f8c97d29da35c0f.png)

```c++
#include <iostream>
using namespace std;
int n,m,q;
int N=1e6+10;
int a[N][N],s[N][N];


int main(){
    cin>>n>>m>>q;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            cin>>a[i][j];
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            s[i][j]=s[i-1][j]+s[i][j-1]-s[i-1][j-1];//前缀和 
    while(q--){
        int x1,y1,x2,y2;
        cin>>x1>>y1>>x2>>y2;
        cout<<s[x2][y2]-s[x2][y1-1]-s[x1-1][y2]+s[x1-1][y1-1]<<endl;
    }
    return 0;
}

```

### 3.一维差分

**差分实质是前缀的逆运算**

> 假设有一个数组，a{1}，a{2}，a{3}，a{4}，a{5}，…，a{n}
>
> 针对这个数组，构造出另一个数组，b{1}，b{2}，b{3}，b{4}，b{5}，…，b{n}
>
> 使得a数组是b数组的前缀和，即使得 a{i} = b{1} + b{2} + … + b{i}；b{i}=a{i}-a{i-1}

> **差分的作用：**
>
> 若要对a数组中[l, r]区间内的全部元素都加上一个常数C，若直接操作a数组的话，需要循环遍历，时间复杂度是O(n)。而如果操作其差分数组b，则时间复杂度是O(1)。**即可以用O(1)的时间给某一数组的某一子序列元素都加上一个值**
>
> **具体实现：**
>
> 数组a是数组b的前缀和数组，只要对` b{l}`这个元素加`C`，则`a`数组从`l`位置之后的全部数都会被加上`C`，但`r`位置之后的所有数也都加了`C`，所以我们通过对 `b{r+1}` 这个数减去`C`，来保持`a`数组中r位置以后的数的值不变，即`a`数组`r`位置以后的数都是`+C-C`抵消。
>
> 于是，对`a`数组的`[l, r]`区间内的所有数都加上一个常数`C`，就可以转变为对` b{l}+C`，对`b{r+1}-C`。
>
> **对于构造差分数组b：**
>
> 在输入数组a时，可以先假想数组a和数组b的全部元素都是0。然后每次进行一次插入操作（指的是对数组a的`[l, r]`区间的每个数加上常数C），比如对a数组区间[1,1]，加（插入）常数a{1}，效果就是` b{1}=b{1}+a{1}`，`b{2}=b{2}-a{1}`；对区间[2,2]，加常数a{2}，效果就是` b{2}=b{2}+a{2}`，`b{3}=b{3}-a{2}`,…，这样在输入数组a的同时，就能够快速构造出其差分数组b。**实际就把构造数组b的过程看作为给一个数组中[l, r]区间内的全部元素都加上一个常数C的过程，只不过特殊的是这个数组元素都为0，这个区间为[i,i]，这个常数C会变化为a[i]**

**AcWing 797.差分**

![img](assets\753022215368408da83b5f20e8d12a6a.png)

```c++
#include <iostream>
using namespace std;
int n,m;
int N=1e6+10;
int a[N],b[N];

void insert(int l,int r,int c){
    b[l]+=c;
    b[r+1]-=c;
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        cin>>a[i];
    for(int i=1;i<=n;i++)
        insert(i,i,a[i]);//构造差分数组b
    while(m--){
        int l,r,c;
        cin>>l>>r>>c;
        insert(l,r,c);
    }
    for(int i=1;i<=n;i++) b[i]+=b[i-1];//构造数组b的前缀和，即最终结果
    for(int i=1;i<=n;i++) cout<<b[i]<<" ";
    
    return 0;
}

```

### 4.二维差分

> 类比一维差分和二维前缀和，**二维差分的作用：以时间复杂度O(1)对任意子矩阵加上一个常数**
>
> 
>
> **具体实现：**
>
> 期望对矩阵a中左上角为[x1, y1]，右下角为[x2, y2]的区域内的全部元素，都加一个常数C，则可以转化为对其差分矩阵b的操作。
>
> 先对b中[x1, y1]位置上的元素加C，这样以来，前缀和a中[x1, y1]这个点的右下角区域内的所有数都加上了C，但是这样就对[x2, y2]之后的区域也都加了C。
>
> 我们对[x2, y2]之外的区域需要保持值不变，所以需要进行减法。对b{x2+1,y1} 减掉C，这样下图红色区域都被减了C，再对b{x1,y2+1}减掉C，这样下图蓝色区域都被减了C，而红色区域和蓝色区域有重叠，重叠的区域被减了2次C，所以要再加回一个C，即对b{x2+1,y2+1}加上一个C。这样，就完成了对[x1, y1]，[x2, y2]区域内的所有数（下图绿色区域），都加上常数C。
>
> ![image-20240714212527225](assets\image-20240714212527225.png)
>
> ![image-20240714211359835](assets\image-20240714211359835.png)
>
> **构造差分二维数组b**
>
> 思想和一维差分一样，假设二维数组a和二维数组b初始化都为0，在输入`a[i][j]`时就可顺便构造`b[i][j]`。**实际就把构造数组b的过程看作为给一个数组中[x1, y1]，[x2, y2]区域内的全部元素都加上一个常数C的过程，只不过特殊的是这个数组元素初始都为0，这个区域为[i,j]到[i,j]（即就是一个元素），这个常数C会变化为`a[i][j]`**
>
> 

**AcWing 798.差分矩阵**

![img](assets\3f287d4deea64d24a6476643b24f74db.png)

```c++
#include <iostream>
using namespace std;
int n,m,q;
int N=1e6+10;
int a[N][N],b[N][N];

void insert(int x1,int y1,int x2,int y2,int c){
    b[x1][y1]+=c;
    b[x2+1][y1]-=c;
    b[x1][y2+1]-=c;
    b[x2+1][y2+1]+=c;
}


int main(){
    cin>>n>>m>>q;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            cin>>a[i][j];
    //构造差分数组b[][]
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            insert(i,j,i,j,a[i][j]);
    while(q--){
        int x1,y1,x2,y2,c;
        cin>>x1>>y1>>x2>>y2>>c;
        insert(x1,y1,x2,y2,c)
    }
    //求差分数组b[][]的前缀和 即加完常数的数组a
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
    		b[i][j]=b[i-1][j]+[i][j-1]-b[i-1][j-1]+b[i][j];
    //输出
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++)
            cout<<b[i][j]<<" ";
        cout<<endl;
    }
    return 0;
}

```



## 5.双指针算法

主要思想：

- 针对单一序列使用两个指针，如快速排序
- 针对不同的两个序列使用两个指针，如归并排序

双指针算法可以实现对暴力（朴素）算法的优化，使时间复杂度由O(n^2)优化到O(n)，要求序列有**单调关系**

**代码模板**

```c++
for(int i=0,j=0;i<n;i++){
    while(j<=i&&check(i,j)) j++;
    //具体逻辑
    .....
}

```



**例：给定一个由单词组成的字符串，单词之间用空格相隔，实现输出各个单词**

实现思路：使用双指针算法，设置指针`i`，`j`，指针`i`指向单词首部，指针`j`向后移找到单词之间的空格。输出`j`和`i`之间的单词，再让`i`后移到`j`的位置，`j`继续后移找到下一个空格，循环直至`j`移动到末尾。

```c++
#include <iostream>
#include <string>
using namespace std;

int main(){
    string str;
    getline(cin,str);//读入空格
    int n=str.length();
    
    for(int i=0;i<n;i++){
        int j=i;
        while(j<n&&str[j]!=' ') j++;
        for(int k=i;k<j;k++) cout<<str[k];
        i=j;
    }
    return 0;
    
}

```

#### **AcWing 799.最长连续不重复子序列**

![img](assets\f462df8e214c4c89a2d946aa83cdbb9f.png)

**实现思路：**

- 暴力算法：两重for循环实现所有元素的枚举，选出满足条件的结果
- **双指针算法**：设置双指针，指针`j`指向结果区间的首元素，指针`i`指向结果的区间的末尾元素。
  - 整数序列用数组`a[]`存储，设置一个计数数组`s[]`，`s[a[i]]`表示元素`a[i]`出现的次数，最后`s[]`中为1的元素就是最终结果区间中的元素。
  - `j`指针首先指向序列首部，`i`指针不断后移，同时对指向的元素进计数。
  - 当出现某个元素计数值>1，即出现重复时，`j`指针指向的元素计数要减1，且`j`指针后移直到此时区间不含重复的数，然后记录当前区间长度，`i`继续后移判断。循环直到`i`指针移动到末尾。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1e6+10;
int a[N],s[N];
int n,res=0;

int main(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
    
    for(int i=0,j=0;i<n;i++){
        s[a[i]]++;
        while(s[a[i]]>1){
            s[a[j]]--;
            j++;
        }
        res=max(res,i-j+1);
    }
    cout<<res;
    return 0;
}


```

**或滑动窗口(leetcode)：**

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
            int n=s.size();
            if(n==0) return 0;
            //使用一个哈希表来表示滑动窗口
            unordered_set<char> mp;
            //maxStr记录最大长度 left指向滑动窗口第一个元素
            int maxStr=0,left=0;
            for(int i=0;i<n;i++){
                //循环寻找不在窗口的元素
                while(mp.find(s[i])!=mp.end()){//若当前元素存在于窗口
                    mp.erase(s[left]);//将左端第一个元素移除窗口
                    left++;//指针后移 窗口缩小 
                }
                //找到的当前元素不存在于窗口中
                maxStr=max(maxStr,i-left+1);//更新最大长度
                //将当前元素插入窗口
                mp.insert(s[i]);
            }
        return maxStr;
    }
};
```



#### AcWing.800 数组元素的目标和

![image-20240721202150304](assets\image-20240721202150304.png)

**实现思路：**

- 暴力算法，直接两重for循环解决，时间复杂度O(n*m)
- 双指针算法，**对于某一个`i`(从左到右遍历)有一个`j`(从右到左遍历)满足`a[i]+b[j]>=x`，且`j`最小**，即为最终的结果。时间复杂度为`O(n+m)`，因为第二重while循环实际上总共只执行m次(j从m减到0)
- 注意：假如不只一个解，则不能使用这个双指针算法

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int n,m;
int a[N],b[N];

int main(){
    int x;
    cin>>n>>m>>x;
    for(int i=0;i<n;i++) cin>>a[i];
    for(int j=0;j<m;j++) cin>>b[i];
    //双指针算法
    for(int i=0,j=m-1;i<n;i++){
        while(j>=0 && a[i]+b[j]>x) j--;
   		if(a[i]+b[j]==x){
       	 cout>>i>>" ">>j;
         break;
    	} 
   }
    return 0;
}

```



## 6.位运算

主要思想：

- **获取一个数的二进制的第k位**：`x>>k & 1`，将x右移k位即**第k位就移动到最低有效位**，每次右移后的数**与1做按位与运算**，只会保留最低有效位即第k位的二进制数。如x=10111，x>>2，得x=101，与1按位与，即101&001=001，最后就得到1

- **获取一个数的二进制的最后一位1开始的二进制数，即10..0**：`lowbit(x)= x & -x`

  > 例如 x = 1010 lowbit(x) = 10；x = 101000 lowbit(x) = 1000
  >
  > lowbit运算的原理是，`x & -x`，由于-x采用补码表示，它等于对x的原码取反再加1，即`-x = ~x + 1`
  >
  > 比如 x 的二进制表示是：100101000，对x取反得011010111，加1得011011000（即-x），经过`x & (~x + 1)`，x的最后一位1，被保留了下来，这个位置后面，两个数全是0；这个位置前面，两个数是取反，做与运算后也全为0，即最终结果就为000001000
  >
  > lowbit的最简单的应用：**统计x的二进制表示中，1的个数**。具体的实现方式是：每次对x做lowbit运算，并将运算结果从x中减去。循环做下去，直到x被减为0，一共减了多少次，x中就有多少个1。

  #### **AcWing801.二进制中1的个数**

  ![image-20240721205921525](assets\image-20240721205921525.png)

  ```c++
  #include <iostream>
  using namespace std;
  
  //找到x最后一个1开始的二进制数
  int lowbit(int x){
      return x & -x;
  }
  
  int mian(){
  	int n;
      cin>>n;
      while(n--){
          int x;
          cin>>x;
          int res=0;
          while(x) x-=lowbit(x),res++;//每次减去x最后一位1
      }
      cout<<res;
      return 0;
  }
  
  ```

  

## 7.离散化

**主要思想：**

有的数组，其元素的值域很大，比如数组中的元素范围很大，下标范围为$[-10^9, 10^9]$，但元素的个数很少，比如只有$1000$个元素。

有时（例如计数排序的思想），**我们需要将元素的值，作为数组的下标来操作**。此时不可能开一个$2*10^9$大小的数组。此时我们把这些元素，**映射到下标从0（或者从1）开始的数组中**。（也可以理解为对稀疏数组进行压缩）

**或者理解为较大的数组下标与较小的数组下标建立映射关系**

> 例如：
>
> 有一个数组a，存储1, 3, 100, 2000, 500000，**此时若需要用该数组中元素值作为数组下标进行某种操作**，这就意味着操作数组下标范范围很大但实际存储数据的位置就那么几个，所以我们把这个数组中的元素，**映射到与[0, 1, 2, 3, 4]自然数相对应**，即用一个新数组all[]，令all[0]=1,all[1]=3....all[4]=500000，这个映射的过程，称之为**离散化**。**实际上离散化的是数组下标值，而不是实际数据，因为要将这些元素值作为下标去操作一个数组**

**离散化要点：**

- 用数组`all`从下标`0`开始存储原数组`a`中的值，要求离散化数组`all`有序，且若有重复元素，可能需要去重(**利用C++中的库函数实现排序和去重**)
- 实现离散化，即得到离散化值（`find`函数返回数组下标），就能代表形成映射关系，元素值与数组下标对应
- 对离散化后的值进行操作：由于数组`all`已经排好序，故这里用二分查找`x`在数组`all`中的位置`q`（`[0~n-1]`，即实现离散化)

**模板**

```c++
vector<int> v;//离散化的数组
sort(v.begin(),v.end());//将数组排序 C++中的sort函数
v.erase(unique(v.begin(),v.end()),v.end());//对数组去重  C++中的unique、erase函数
//unique函数将数组v的全部重复元素删掉，再把不重复元素全部放到数组前面，返回新数组的最后一个位置，然后用erase函数将这个位置到最后一个位置的元素删掉

//查找数x在数组中的位置即下标 二分查找，实际上就是x对应的离散化值（映射到0/1开始的自然数）
int find(int x){
    int l=0,r=v.size()-1;
    while(l<r){
        int mid=(l+r)/2;
        if(v[mid]>=x) r=mid;
        else l=mid+1;
    }
    return r+1;//l+1也可，这里是否加1和题目有关，若题目设计前缀和差分，要从1开始，则需要加1
    
}

```

#### **Acwing 802.区间和**

![img](assets\6817abba667b4ce3bb8f631a94ba8f3a.png)

实现思路：

- 先读入插入操作需要的数字`(x,c)`将其存储在`add`数组（实现加数操作）当中，同时将`x`压入`alls`数组当中。再读入问询操作需要的数字`(l,r)`将其存储在`query`数组（实现查询区间和操作）当中，同时将`l,r`压入`alls`数组当中（**注意这里想`x,l,r`都代表坐标轴位置**）。
- 再对数组`alls`进行排序，去重操作，运用c++自带的sort函数对alls数组进行排序，再使用erase函数和unit函数对其进行去重操作。
- 再处理加数操作，利用二分查找，得到`x`在`alls`数组的位置`q`（注意加1，因为后续要算区间和即前缀和，数组从1开始），再创建一个数组`a`,**`a[q]`就与坐标位置相对应**（初始值为0），`a[q]+c`就表示对应坐标位置`+c`。
- 再求`a`数组前缀和数组，即`s[i]=s[i-1]+a[i]`，循环终止条件为`i=alls.size()`(因为二分查找最后一个位置加1)。
- 再求区间和，同加数操作，利用二分查找，得到`l,r`在`alls`数组的位置，通过前缀和数组求出区间和（`s[r]-s[l-1]`）。

**其他理解**：

[AcWing 802. 区间和 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/1783/)

http://t.csdnimg.cn/XPtTa

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

typedef pair<int,int> PII;//用于存储对应数，如x和c，l和r

const int N=3e6+10;
vector<int> alls;//需离散化数组
int a[N],s[N];//数组a代表坐标轴上的数，进行加数操作，并求前缀和数组s
int n,m;
vector<PII> add,query;//x和c，l和r

//形成离散化值 得到x对应下标位置 注意加1，因为后续要求前缀和
int find(int x){
    //二分查找x的位置
    int l=0,r=alls.size()-1;
    while(l<r){
        int mid=(l+r)/2;
        if(alls[mid]>=x) r=mid;
        else l=mid+1; 
    }
    return r+1;
}

int main(){
    cin>>n>>m;
    while(n--){
        int x,c;
        cin>>x>>c;
        add.push_back({x,c});
        alls.push_back(x);//同时将x加入需离散化数组 x代表坐标位置
    }
    while(m--){
        int l,r;
        cin>>l>>r;
        query.push_back({l,r});
        //同时将l，r加入离散化数组 两者都代表坐标位置
        alls.push_back(l);
        alls.push_back(r);
    }
    //排序
    sort(alls.begin(),alls.end());
    //去重
    alls.erase(unique(alls.begin(),alls.end()),alls.end());
    
    //取离散化值完成对应操作
    for(auto item:add){
        int q;
        q=find(item.first);//得到坐标位置x对应的离散化值
        a[q]+=item.second;//坐标轴x位置加c
    }
    //计算前缀和
    for(int i=1;i<=alls.size();i++) s[i]=s[i-1]+a[i];    
    
    //输出区间和
    for(auto item:query){
        int l,r;
        l=find(item.first);
        r=find(item.second);
        cout<<s[r]-s[l-1]<<endl;
    }
    return 0;
}


```



## 8.区间合并

**主要思想：**给定很多个区间，若2个区间有交集，将二者合并成一个区间

- 先按照区间的左端点进行排序
- 然后遍历每个区间，根据不同情况进行合并，对于两个区间（后一个区间为`i`）可能有下面几种关系：

![Image](assets\e706e67)

> - 对于第一种情况，区间不变
> - 对于第二种情况，`end`要变成区间`i`的右端点
>   - 前面两种情况，可以合并为将`end`更新为`end`和区间`i`的右端点中的较大者
> - 对于第三种情况，将当前维护的区间加入答案，并将维护的区间更新为区间`i`

#### **Acwing 803.区间合并**

![img](assets\933faa0d1ec24e77aaebbfd1d14d1016.png)

**实现思路：**

- 先读入区间端点`(l,r)`，将其存入`segs`数组内，再设置一个结果数组`res`
- 将`segs`数组按左端点排序，设置左右端点`（st,ed）`为第一个维护的区间，初始都为负无穷大
- 依次取出`segs`数组中的元素，比较当前区间（维护区间）`ed`和另一个区间`seg`的`l`关系，如果`ed<l`，叫说明两个区间没有交集，则直接将`seg`加入`res`中，并将待维护的区间变为`seg`（即后移，设置新的维护区间）。反之，更新`ed`为最大值（因为区间排好序，所以不用将`st`更新为最小）。
- 最后判断`segs`是否为空，若不为空，则将最后剩余一个维护的区间加入`res`中。

```c++
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

typedef pair<int,int> PII;
vector<PII> segs;
int n;

void merge(vector<PII> &segs){
    vector<PII> res;//设置一个结果数组
    int st=-2e9,ed=-2e9;//用-2*10^9代表负无穷
    for(auto seg:segs){
        if(ed<seg.first){//当前维护区间与下一个区间无交集
            if(st!=-2e9) res.push_back({st,ed});//判断初始化区间负无穷
            st=seg.first,ed=seg.second;//后移，更新维护区间
        }else ed=max(ed,seg.second);//有交集，取两个区间的右端点最大值
    }
    if(st!=-2e9) res.push_back({st,ed});//判断，然后将剩下的区间加入结果
    segs=res;
}

int main() {
    cin>>n;
    while(n--)
    {
        int l,r;
        cin>>l>>r;
        segs.push_back({l,r});
    }
     //先对区间按左端点排序
    sort(segs.begin(),segs.end());//sort自动对pair先按左端点排序，再按右端点排序
    merge(segs);
    cout<<segs.size();
    return 0;
}

```



# 第二章 数据结构

## 1.链表

**主要思想**：使用**数组实现**链表（**而不用结构体，结构体代码更长，后续图论也是基于数组实现**），即**静态链表**。因为动态链表使用new申请空间需要较多的时间，而算法要求的是以较少的时间完成任务。

1. 单链表，最主要用单链表写邻接表，用邻接表存储图或者树
2. 双链表，优化某些问题

### 1）单链表

**单链表：**头指针head指向第一个结点，初始为-1。数组e[]存储结点的值，数组ne[]存储对应结点的下一个结点的下标，形成一个链条。**本质上是用数组下标来操作对应结点**

#### AcWing 826.单链表

![img](assets\0dd51d02fe4f4a5a9cbb31853bf4bd70.png)

**实现思路：**

- 设置数组`e[]`存储结点值，数组`ne[]`存储对应结点下一个结点的下标

- 设置头指针`head`指向第一个结点，初始值为`-1`

- 设置指针`idx`表示当前操作位置，初始为`0`即指向数组的第一个位置，单增（**实际上`idx`所指向的位置就是将要操作的数组位置，这个位置还未存入结点，每次存入结点后`idx`需要后移，但删除结点时不用考虑当前数组位置是否浪费而做出某种处理，算法以速度为主，因此删除结点是`idx`结点不用前移**）

  

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;

int e[N],ne[N];
int head,idx;

//初始化
void init(){
    head=-1;
    idx=0;
}

//链表头插入结点
void add_to_head(int x){
    e[idx]=x;
    ne[idx]=head;
    head=idx;//根据数组表示链表的性质，随着链表头不断插入结点，第一个结点的位置是后移的，即head后移到idx
    idx++;//插入结点后idx指针后移
}
//在下标为k的结点后面插入结点
void add(int k,int x){
    e[idx]=x;
    ne[idx]=ne[k];
    ne[k]=idx;
    idx++;
}

//删除下标为k的结点后面的结点
void remove(int k){
    if(k==-1) head=ne[head];//即删除头结点
    else ne[k]=ne[ne[k]];
}
int main() {
    int m;
    cin>>m;
    init();
    while(m--)
    {
        char s;
        cin>>s;
        if(s=='D')
        {
            int k;
            cin>>k;
          	remove(k-1);
        }
        else if(s=='H')
        {
            int x;
            cin>>x;
            add_to_head(x);
        }
        else
        {
            int k,x;
            cin>>k>>x;
            add(k-1,x);
        }
    }
    for(int i=head;i!=-1;i=ne[i]) cout<<e[i]<<' ';
    return 0;
}


```

### 2）双链表

**双链表：**此时没有头指针，固定数组前两个位置的存储，下标`0`的位置存储链表第一个结点（即左端点），数组下标`1`的位置存储链表的末尾结点（即右端点）。三个数组，数组`e[]`存储结点值，数组`l[]`指向当前结点的左结点（即前驱），数组`r[]`指向当前结点的右结点（即后继），`idx`指向当前操作位置.

#### AcWing 827.双链表

![img](assets\e8c9ba6642084d32969ac4fb2603c1fd.png)

**实现思路**：类似单链表

- 注意`idx`初始为2，因为数组前两个位置已固定存储链表首尾结点
- 初始左端点右指向右端点，右端点左指向左端点

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;

int e[N],l[N],r[N];
int idx;
int n;

//初始化
void init(){
    r[0]=1;
    l[1]=0;
    idx=2;
}

//在下标为k的结点的右侧插入结点   若是左侧插入结点则传入的k为l[k]
void add(int k,inx){
    e[idx]=x;
    l[idx]=k;
    r[idx]=r[k];
    l[r[k]]=idx;
    r[k]=idx;
    idx++;
}
//删除下标为k的结点
void remove(int k){
    r[l[k]]=r[k];
    l[r[k]]=l[k];
}
int main()
{
    cin>>n;
    init();
    while(n--)
    {
        string op;
        cin>>op;
        if(op=="L")
        {
            int x;
            cin>>x;
            add(0,x);
        }
        else if(op=="R")
        {
            int x;
            cin>>x;
            add(l[1],x);
        }
        else if(op=="D")
        {
            int k;
            cin>>k;
            remove(k+1);
        }
        else if(op=="IL")
        {
            int k,x;
            cin>>k>>x;
            add(l[k+1],x);
        }
        else 
        {
            int k,x;
            cin>>k>>x;
            add(k+1,x);
        }
    }
    for(int i=r[0];i!=1;i=r[i]) cout<<e[i]<<" ";
}



```



## 2.栈与队列

### 1）数组模拟栈

主要思想：**先进后出**，设置一个数组存储数据，一个栈顶指针指向栈顶（初始化为0）

注意：数据出栈，只是指针单纯前移，无需在意数组还存储数据造成浪费

#### AcWing 828.模拟栈



![img](assets\64a1619f136443a49a97e8c1f7486238.png)

```c++
#include <iostream>
#include <string>
using namespace std;
const int N=1000010;
int st[N],tt;
int main()
{
    int n;
    cin>>n;
    while(n--)
    {
        string op;
        cin>>op;
        if(op=="push")
        {
            int x;
            cin>>x;
            st[++tt]=x;
        }
        else if(op=="pop")
        {
            tt--;
        }
        else if(op=="query")
        {
            cout<<st[tt]<<endl;
        }
        else
        {
            if(tt>0) puts("No");
            else puts("Yes");
        }
    }
    return 0;
}
```



### 2）数组模拟队列

主要思想：**先进先出**，队尾插入，队头删除。设置一个数组存储数据，队头指针指向队列第一个元素（初始为0），队尾指针指向最后一个元素（初始为-1）

#### AcWing 829.模拟队列

![img](assets\a764131a56a34ddcade2860e52f5b867.png)

```c++
#include <iostream>
#include <string>
using namespace std;
const int N=1000010;
int queue[N],hh,tt=-1;
int main()
{
    int n;
    cin>>n;
    while(n--)
    {
        string op;
        cin>>op;
        if(op=="push")
        {
            int x;
            cin>>x;
            queue[++tt]=x;
        }
        else if(op=="pop")
        {
            hh++;
        }
        else if(op=="query")
        {
            cout<<queue[hh]<<endl;
        }
        else
        {
            if(hh<=tt) puts("No");
            else puts("Yes");
        }
    }
    return 0;
}
```



### 3）单调栈

**主要应用**：对一个序列中的某个数，输出这个数左边第一个比他小的数

#### AcWing 830.单调栈

![img](assets\41bc641b2aed4620b815e3c25e6341e1.png)

**实现思路：**

- 若采用暴力算法，直接两重循环逐个遍历比较，时间复杂度为O(n^2)。
- 采用构造单调栈的思想，降低时间复杂度为O(2n)。每次输入一个数据就判断它左边第一个比他小的数
  - 比如对于序列`[3, 4, 2, 7, 5]`，求解每个数左边最近的且比它小的数（不存在则返回-1）。对于第`i`个元素，假设`j < k < i`如果`a[k] < a[j]`， 那么`a[j]` 绝对不会是`i`的答案（**即后来者比前面的小，则答案只会在后来者或者更后来的更小的数中选出，而不会在前面的数中得到，那么这个前面的数就可以除去**）。以此构造一个栈，当后来者小于栈顶元素时，栈顶元素不断出栈（**弹出栈的元素已失去角逐后续输出答案的机会，后来者居上**），然后当前元素入栈（在栈顶），这样就构造了一个**单调递增栈**。
- 用数组构造一个栈，栈顶指针为`tt`。每次输出一个数据就将他与当前栈顶元素比较。
  - 若栈不为空且栈顶元素大于当前数据，弹出栈，指针`tt`前移。若直到栈空，则左边不存在比当前元素小的数，输出`-1`；
  - 若找到比当前元素的小的栈顶元素，即为答案输出栈顶元素
- 最后要将当前数入栈，因为当前元素也可能是后来者的答案

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int stk[N],tt=0,n;
int main(){
    scanf("%d",&n);
    while(n--){
        int x;
        scanf("%d",&x);
        while(tt && stk[tt]>= x) tt--;
        if(tt) printf("%d ",stk[tt]);//当前数左端存在比当前数小的数
        else printf("-1");//不存在
        stk[++tt]=x;//当前数入栈
    }
    return 0;
    
}
```



### 4）单调队列

主要应用：求滑动窗口中的最小、最大值

#### AcWing 154. 滑动窗口

![img](assets\8151e7d65eb54f8c8d93e89c2feb832c.png)

![img](assets\912142c67c2c4aa5b75614a8c19f11b8.png)

**实现思路**：**以输出窗口最小值**为例，考虑暴力算法，每滑动一次窗口，都比较`k`次，寻找到最小值，注意到如果左侧的数据大于右侧的数据，则可以在下一次滑动前删除左侧的数据，以此减少比较的次数，就能构造一个**单调递增的队列**，每次只需输出队头元素即为窗口内的最小值

- 设置一个滑动窗口队列(队头在左，队尾在右)，比较队尾和当前即将进入滑动窗口的数字大小，如果队尾数字大，则删除队尾。因此该队列中的数字全都小于待插入的数字，并且队头是最小的数字，再将该元素插入到队尾，最后输出队头即可。
- 使用一个**`q[]`数组**表示该队列，`hh`指向队头，`tt`指向队尾前一个位置，**注意该数组存储的是对应元素的下标而不是实际元素值**。
- **维持当前窗口的长度为`k`，若长度大于`k`，队头指针后移，表示窗口缩小，直到队头指针和`i`之间的长度为`k`**
- 循环比较待插入元素和队尾的关系，**两种情况**
  - ①队列不为空且待插元素一直比队尾元素小，则队尾元素不断出队。若直到队列为空，退出当前循环，让待插入元素存入队尾（此时队列仅一个元素）
  - ②若出现待插入元素比队尾元素大，那直接退出循环，让待插入元素存到队尾
  - **最终就使得队列中的元素是递增的**
- 当窗口中的元素已满足`k`个，满足输出条件，输出最小值，即队头`q[hh]`所指元素；最大值即队尾元素`q[tt]`。

**其他理解**：

[AcWing 154. 滑动窗口---海绵宝宝来喽 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/4139707/)

```c++
#include <iostream>
using namespace std;
const int N= 1e6+10;
int a[N],q[N];
int main(){
    int n,k;
    cin>>n>>k;
    for(int i=0;i<n;i++) cin>>a[i];
    
    //输出窗口内最小值
    int hh=0,tt=-1;//队头队尾指针初始化
    for(int i=0;i<n;i++){
        //维持窗口长度 大于队头指针指向的下标则 队头指针后移
        while(hh<=tt && i-k+1>q[hh]) hh++; //这里用if也行 因为hh只会加一次
        //使队列元素单增
        while(hh<=tt && a[q[tt]]>=a[i]) tt--;
        q[++tt]=i;//存的是下标
        //窗口内已有k个值，可以输出最小值
        if(i>=k-1) cout<<a[q[hh]]<<" ";
    }
    cout<<endl;
    
    
    //输出窗口内最大值
    hh=0,tt=-1;//队头队尾指针初始化
    for(int i=0;i<n;i++){
        //先判断队头元素是否还在窗口内
        if(hh<=tt && i-k+1>q[hh]) hh++;
        //使队列元素单减
        while(hh<=tt && a[q[tt]]<=a[i]) tt--;
        q[++tt]=i;//存的是下标
        //窗口内已有k个值，可以输出最大值
        if(i>=k-1) cout<<a[q[hh]]<<" ";
    }
    return 0;
}

```



## 3.KMP

> KMP算法主要用在**字符串匹配**上。
>
> 比如我们从字符串"acfacfgded"（需要在哪里找的字符串称为“文本串”）找其中是否包含字符串"acfg"（需要从文本串里找的字符串我们叫做“**模式串**”），我们一般会想到的解法是暴力求解，两层for循环，依次对模式串的每一个元素进行匹配，如果匹配失败，下次还从模式串的第一个进行匹配，这就导致了较高的时间复杂度（O(n×m)）。
>
> 而KMP算法不同之处就在于，**当模式串的某个元素匹配失败后，不需要再从模式串的第一个元素从头开始匹配了，而是根据前缀表(next)找到模式串中一个最优的位置继续进行匹配**

1.**前缀表（next数组）：next[i] 记录的是模式串下标 i（包括i）之前的字符串的最长相等前后缀的长度**。规定next[0]=0，因为第二个字符就不匹配，自然就回到第一个字符开始匹配

对于字符串"acdac"

![image-20240719173015089](assets\image-20240719173015089.png)

另一种理解：**将一个字符串固定不动，另外一个完全一样的字符串不断向右平移，直到两个字符串的交叉部分的元素相等，相等前后缀就是两个字符串的交叉部分**

![image-20240719173114154](assets\image-20240719173114154.png)

- **关于next数组的代码求解**，思想类似kmp匹配两个字符串，特殊的只是**将自己同时当作文本串和模式串进行匹配**

2.得到前缀表后，设指向文本串的指针为`i`，指向模式串的指针为`j`，初始都为0；文本串数组为s[]，模式串数组为p[]，next数组为ne[]。文本串 “aabaabaafa”，模式串 “aabaaf” 。

- 开始循环比较s[i]与p[j]，当出现`j>0且s[i]!=p[j]`时

![image-20240719175104896](assets\image-20240719175104896.png)

- 利用前缀表，让模式串向右平移一段位置（而不是像暴力那样直接模式串后移一位重新开始比较），找到文本串和模式串匹配的前后缀，就是改变指针j的位置，**令`j=ne[j-1](j>0)`，即让当前指针值等于前一个字符对应的前缀值**，不用再从头开始匹配。这里就让`j=ne[4]=2`，即指向b

![image-20240719175709324](assets\image-20240719175709324.png)

#### AcWing 831.KMP字符串匹配

![img](assets\bd06f9beb039475da0f19dd79c9cb17b.png)

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
char s[N],p[N];//文本串和模式串
int n,m;
int ne[N];//next数组

int mian(){
    cin>>n>>p>>m>>s;
    
    //得到next数组
    for(int i=1,j=0;i<n;i++){//i从1开始，因为ne[0]=0
        while(j>0 && p[i]!=p[j]) j=ne[j-1];//要保证j>0 出现不匹配
        if(p[i]==p[j]) j++;//匹配 则后移继续
        ne[i]=j;//j就是代表已经匹配部分的长度
    }
    
    //进行kmp匹配
    for(int i=0,j=0;i<m;i++){
        while(j>0 && s[i]!=p[j]) j=ne[j-1];//要保证j>0 出现不匹配
        if(s[i]==p[j]) j++;//匹配 则后移继续
        if(j==n){//匹配结束
            printf("%d",i-n+1);
        }
    }
}

```



## 4.Trie树（字典树）

Trie树，又称字典树，是用来**高效存储和查找字符串集合**的一种数据结构查找时，可以**高效的查找某个字符串**是否在Trie树中出现过，并且可以查找出现了多少次

利用字符串的**公共前缀**来减少查询时间，最大限度地减少无谓的字符串比较，查询效率比哈希树高。

![image-20240720093502118](assets\image-20240720093502118.png)

**主要性质：**

- **根节点不包含字符，除根节点外的每一个子节点都包含一个字符。**
- 从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。

- 每个节点的所有子节点包含的字符互不相同。

- 从第一字符开始有连续重复的字符只占用一个节点，比如上面的catch和cat中重复的单词cat只占用了一个节点。



#### Acwing 835.Trie字符串统计

![img](assets\8ec17ddbc4d44066b46df3b237a6c585.png)

实现思路：

- 设置一个二维数组`s[p][u]`初始为0，`p`代表当前结点，`u`代表当前结点的某个子结点（u的取值为0~25对应26个字母），即`s[p][u]`不为0代表`p`结点下连接着下标为`u`的字母
- 插入操作：设置一个`idx`指向要操作的数组下标位置（作用同单链表中的idx），每次插入一个字符串，循环处理每一个字符。利用`u=str[i]-'a'`将每个字符转化为整数操作，判断`s[p][u]`的值是否为0。若为0，不存在该字符，就插入该字符，即令`s[p][u]=++idx`，然后下移到子结点继续插入字符串的下一个字符；若不为0，就表示之前已经插入过该字符，直接下移。最后该字符串插入结束，设置一个数组`cnt[]`标记以该字符串出现次数，`cnt[p]`表示以`p`结尾的字符串出现次数。
- 查询操作：类似插入操作进行判断，若出现某一次`s[p][u]`为0就此结束，意味着不存在该字符串，直到循环结束，存在该字符串，返回计数数组`cnt[p]`

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;

char str[N];
int s[N][26],cnt[N],idx;
int n;

//插入一个字符串
void insert(char[] str){
    int p=0;//树的指针，用于下移，初始指向根节点 根节点不存储字符
    int u;//存储当前判断字符的整数值（即数组下标）
    for(int i=0;str[i];i++){//字符数组以'\0'结尾，即每次判断str[i]是否为空字符'\0';或#include <cstring> 使用函数strlen(str)得到字符数组长度
        u=str[i]-'a';//映射到0~25
        if(!s[p][u]) s[p][u]=++idx;//如果不存在，插入该字符
        p=s[p][u];//指针下移
    }
    cnt[p]++;//字符串计数
}

//查询字符串出现次数
int query(char[] str){
    int p=0,u;
    for(int i=0;str[i];i++){
        u=str[i]-'a';
        if(!s[p][u]) return 0;//某个字符不存在直接退出
        p=s[p][u];
    }
    return cnt[p];
}

int main() {
    cin>>n;
    while(n--)
    {
        char o;
        cin>>o>>str;
        if(o=='I') insert(str);
        else cout<<query(str)<<endl;
    }
    return 0;
}


```

#### Acwing 143. 最大异或对

![img](assets\249278e79bbd4812a5c7dbac7e734088.png)

**实现思路：**异或运算：x^y

1. 暴力算法，两重for循环
2. **Trie树**：将每个整数转化为二进制数，每一位作为结点建立Trie树。从根节点开始记录每个数的二进制最高位。

![QQ图片20240721230341](assets\QQ图片20240721230341.jpg)

- 求**整数x的二进制数**：`x>>k &1`

- **对于某个数，在集合中找到与之异或值最大的数**：由异或的计算可知，**从高位开始，每一位都与当前数不同**，即异或值为1时，这个数越可能是与其异或最大的数。由此对于某个数，只要**尽量选择Trie树自上而下每一步路径上结点与当前数二进制位不同的路径**，最终走到叶子结点时就为与当前数异或值最大的数。

- 选择**先插入再查询**的方法（不先完全建立起Trie树，而是**逐步插入查找然后建立**，这就避免了循环中先异或了`a[i]`与`a[j]`，后面又异或了`a[j]`与`a[i]`；相比先查找再插入也避免了一开始的判空操作）
  - 从第一个数据开始，Trie树为空，直接插入数据到Trie树；第二个数据，插入Trie树，再查找Trie树，找到与第二个数据异或值最大的数；第三个数，插入Trie树，再查找Trie树，找到与第三个数据异或值最大的数....以此类推，每次存下异或值，比较得到最大异或值，最后输出
- 本题数据最大为$2^{31}$，因此每个数的二进制位可以设置最高位从30开始，到0结束。
  - 设置一个**二维数组`s[p][u]`**，表示下标为p的数的子结点u的值，**若值为0表示当前要插入节点不存在，则插入该结点；若不为0则表示当前要插入的结点已存在，下移继续插入**。p范围为所有数据总共的二进制位数，**u**范围为2，即两条路径（子结点）**取值为1或0**。设置整型变量idx，用来当做`s[p][u]`的值，每次插入后自增。设置p指针向下遍历Trie树，每次下移`p=s[p][u]`。
  - 插入结点：循环得到每个数的二进制位，从最高位第30位开始插入Trie树
  - 查找与当前数异或值最大的数：从最高位开始判断选择路径，看与当前二进制位不同的是否有路径，即`s[p][!u]`是否为0，若不为0则有不同的二进制位路径，选择这个路径继续向下遍历；若为0，则表示只有相同的二进制位的路径可走，不得不选择这个路径继续遍历。
  - 每次选择路径后，都要记录当前路径，用一个整型变量res，每次选择路径后`(res*2)+u（或+!u）`，*2即左移1位因为自上而下是高位开始，再加上当前路径值。最终res即为与当前异或值最大的数。
  - 最后比较每次异或值res，选出最大的值输出

```c++
#include <iostream>
#include <algorithm>
const int N=1e5+10,M=31*N;//每个数31位二进制位，共N个数
int n;
int a[N];
int s[M][2],idx;//s数组第二维就存两个

void insert(int x){
    int p=0;//结点指针
    for(int i=30;i>=0;i--){
     int u=x>>i & 1; //计算二进制位，从最高位开始
     if(!s[p][u]) s[p][u]=++idx;//当前位置子结点为空 可插入
     p=s[p][u];//指针后移 继续插入
    }
}

//寻找与x异或值最大的数
int query(int x){
    int p=0,res=0;
    for(int i=30;i>=30;i--){
        int u=x>>i & 1;
        if(s[p][!u]){//如果存在当前二进制位不同的路径
            p=s[p][!u];//指针下移
            res=res*2+!u;
        }else{//只有相同二进制位的路径 那不得不走
            p=s[p][u];
            res=res*2+u;
        }
    }
    return res;
}


int main()
{
    int res=0;
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
    for(int i=0;i<n;i++) 
    {
        //先插入
        insert(a[i]);
        //后查找
        int t=query(a[i]);
        res=max(res,a[i]^t);
    }
    cout<<res;
    return 0;
}

```





## 5.并查集

并查集结构能够支持快速进行如下的操作：

1. 将两个集合合并
2. 询问两个元素是否在一个集合当中

**并查集可以在近乎O(1)的时间复杂度下，完成上述2个操作**

**并查集的基本原理**：用树的形式来维护一个集合。用**树的根节点来代表这个集合的编号**。对于树中的每个节点，都用一个数组存储其父节点的编号。比如一个节点编号为`x`，我们用`p[x]`来表示其父节点的编号。当我们想求某一个节点所属的集合时，找到其父节点，并一直往上找，直到找到根节点

**1.如何才能判断根结点？**

当父节点为其本身时，此结点即为根节点：`p[x]=x`

**2.如何求某结点x的集合编号？**

从当前结点的父节点不断向上遍历，直到找到根节点，即为集合编号:`while(p[x]!=x)  x=p[x]`;

**3.如何合并两个集合？**

令某个集合的根节点为另一个集合根节点的父节点

**优化：**

未优化前查找某个元素的所属集合（也即查找两个元素是否在一个集合）的时间复杂度为O(logn)即树的高度，为使时间复杂度接近O(1)，采用**路径压缩**的方法优化，**即每次查找元素所属集合的过程中，顺便使每个结点的父节点都指向根节点（即集合编号结点）**

**查询根节点（即所属集合编号）：find(x)**

**合并两个集合：p[find(a)]=find(b);**

**判断两个元素是否在同一个集合：find(a)==find(b)**

#### AcWing 836.合并集合

![img](assets\73fc007b0be246bdb3822c9a1968c9f8.png)



```c++
#include <iostream>
using namespace std;

const int N=1e6+10;
int p[N];//父节点数组
int n,m;

//查找某个元素所属集合编号
int find(int x){
    if(p[x]!=x) return p[x]=find(p[x]);//递归查找 同时令每个结点的父结点都指向根节点
    return p[x];
}

int main() {
    cin>>n>>m;
    for(int i=1;i<=n;i++) p[i]=i;//初始化各个元素为一个集合
    while(m--)
    {
        char o[2];
        int x,y;
        cin>>o>>x>>y;
        if(o[0]=='M')
        {
            p[find(x)]=find(y);
        }
        else
        {
            if(find(x)==find(y)) puts("Yes");
            else puts("No");
        }
    }
    return 0;
}


```

#### AcWing 837.连通块中点的数量

![img](assets\1a169abfdfc6475bb20cd42a9717bd75.png)

**实现思路：**一个连通块视为一个集合，相比朴素并查集，就是多了一些维护信息

- 在a和b之间连一条边，即合并集合，若同属一个集合就无需操作
- 询问a和b是否在同一个连通块，即询问a和b是否在同一个集合
- 询问a所在连通块的数量，即每次合并需额外维护一个数组记录集合中的元素个数（只有根节点的记录有意义）

```c++
#include <iostream>
using namespace std;

const int N=1e6+10;
int p[N];//父节点数组
int n,m;
int size[N];//记录对应集合中元素个数 只有集合根节点对应记录有意义

//查找某个元素所属集合编号
int find(int x){
    if(p[x]!=x) return p[x]=find(p[x]);//递归查找 同时令每个结点的父结点都指向根节点
    return p[x];
}

int main() {
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        p[i]=i;//初始化各个元素为一个集合
    	size[i]=1;
    }
    while(m--)
    {
        char o[5];
        int x,y;
        cin>>o>>x>>y;
        if(o[0]=='C')
        {
            if(find(x)==find(y)) continue;
            size[find(y)]+=size[find(x)];
            p[find(x)]=find(y);
        }
        else if(o[1]=='1')
        {
            if(find(x)==find(y)) puts("Yes");
            else puts("No");
        }else{
            cin>>x;
            cout<<size[find(x)]<<endl;
        }
    }
    return 0;
}


```



#### AcWing 240.食物链

![img](assets\c455f027956e4cec9aa9997c706a9240.png)

![img](assets\4304e37dc2834f3aa9ebf0d9f8cff124.png)

实现思路：

- 由题意可知任意两个动物可能存在三种关系之一：被吃，吃，同类。**给定两个动物x和y，若知道这个两个动物与第三个动物z（中间人）的关系，则可推导出x和y的关系**。如x吃y，y吃z，则可得z吃x，要形成一个环形食物链。

- 考虑维护一个**并查集**，任意两个结点（动物）可能是三种关系之一，而只要存在一个中间人，知道这两个结点与中间人的关系就可以推断出这两个结点之间的关系。因此选择并查集的**根结点（集合编号）为中间人**，**通过维护额外的信息来表示各结点与根节点的关系，就可以推断出集合中任意两个结点之间的关系**。
  - 这个额外的信息选择**当前结点到根节点的距离**，因为存在三种关系，则用**距离mod 3**就可以得到三种结果，以此来表示当前结点与根节点的关系。
  - 设定mod 3=0表示当前结点与根节点是同类，mod 3=1表示吃根节点，mod 3=2表示被根节点吃。**则mod 3=2的结点就吃mod 3=1的结点，mod 3值相同的结点是同类**
  - **求每个结点到根节点的距离**：在并查集的路径压缩过程中即设置每个结点的父节点为根节点时来完成。**设置一个距离数组d[]，d[x]表示当前结点与父节点的距离**。首先查找当前结点的根节点记录下来，然后**该结点到根节点的距离=该结点到父节点的距离d[x]+父节点到根节点的距离d[p[x]]，再赋给d[x]，那么d[x]就更新成为当前结点到根节点的距离**。最后更新x的父节点p[x]为根节点。

**对一句话进行判断假话 还是真话**

- 若给出的x、y大于编号最大值n，直接判断为假话
- 否则，继续判断。先得到x和y的根节点。再根据两种说法分别判断
  - **对第一种说法：x和y是同类**。进行判断。若在同一个集合，用两者到根节点的距离对3取模判断，**取模的值不相等就不是同类，为假话**；否则为真话，就要先合并到同一个集合，然后更新x根结点到y根节点的距离使两者到根节点的距离满足x和y是同类。
    - 判断：`d[x]%3!=d[y]%3`====》`(d[x]-d[y])%3!=0` 假话
    - 更新距离（假设x合并到y的集合中）：`(d[x]+d[p[x]])%3=d[y]%3`=====》`d[p[x]]=d[y]-d[x]`
  - **对第二种说法：x吃y。**若在同一个集合，**判断x到根节点的距离mod 3是否满足比y到根节点的距离mod 3大1，若不满足，则x吃y是假话**；否则真话，就要先合并到同一个集合，然后更新x父节点到根节点的距离，满足x吃y是真话。
    - 判断：`d[x]%3-d[y]%3!=1`====》`(d[x]-d[y]-1)%3!=0` 假话
    - 更新距离（假设x合并到y的集合中）：`(d[x]+d[p[x]])%3-d[y]%3=1`====》`d[p[x]]=d[y]-d[x]+1`

**其他理解**：

[AcWing 240. 食物链---数组d的真正含义以及find()函数调用过程 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/1080346/)

```c++
#include <iostream>
using namespace std;
const int N=50010;
int p[N],d[N];//p表示父节点，d表示到父节点的距离 路径压缩后表示到根节点的距离
int n,m;

//得到根节点 并进行路径压缩更新d[x]为节点到根节点的距离
int find(int x){
    if(p[x]!=x){
        int t=find(p[x]);//得到根节点
        d[x]+=d[p[x]];//更新距离为到根节点的距离
        p[x]=t;//更新父节点为根节点
    }
    return p[x];
}
    
int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) p[i]=i;
    int res=0;
    while(m--){
        int q,x,y;
        cin>>q>>x>>y;
        if(x>n || y>n) res++;//假话
        else{
            int px=find(x),py=find(y);//得到各自根节点
            if(q==1){
                if(px==py && (d[x]-d[y])%3!=0) res++;//假话
            	else if(px!=py){//合并集合 更新距离
                	p[px]=py;
                	d[px]=d[y]-d[x];//满足x和y是同类
            	}
            }else{
                if(px==py && (d[x]-d[y]-1)%3!=0) res++;//假话
                else if(px!=py){//合并集合 更新距离
                    p[px]=py;
                    d[px]=d[y]-d[x]+1;//满足x吃y
                }
            }
        }
    }
    cout<<res;
    return 0;
}
```





## 6.堆

堆的基本操作（以小根堆为例，大根堆相反）

1. 插入一个数
2. 求集合当中的最小值
3. 删除最小值
4. 删除任意一个元素
5. 修改任意一个元素

基本结构是一颗**完全二叉树**。以小根堆为例，每个节点都要小于其左右两个子树种的所有节点，**根节点即为集合中的最小值**。

用数组来模拟存储一颗二叉树，采用二叉树层序遍历的方式作为数组的下标。若数组下标从0开始，若某个节点下标为`x`，则其左儿子下标为`2x + 1`，其右儿子下标为`2x + 2`。若数组下标从1开始，若某个节点下标为`x`，则其左儿子下标为**`2x`**，右儿子下标为**`2x + 1`**。**此处讨论数组下标从1开始**

以上所有操作都可以用**下沉down**和**上升up**两个操作来完成

- **down(x)：**注意传入的是数组下标。将`x`与其子结点进行比较，与两个子结点中最小且比`x`小的结点位置进行交换，不断下沉，直到合适的位置
- **up(x)：**注意传入的是数组下标。若`x`比父节点更小，则与父节点位置进行交换，不断上升到合适的位置

**1.插入一个数**

插入到数组末尾，对于这个新插入的数，使用up()不断向上调整

**2.求集合当中的最小值**

即根节点，数组首元素

**3.删除最小值**

交换堆顶和堆尾，然后堆的大小减一。即数组末尾元素覆盖数组首元素，然后对新的堆顶元素进行down向下调整

**4.删除任意一个元素**

交换当前元素和堆尾，然后堆的大小减一。即数组末尾元素覆盖当前元素，然后比较当前元素与被删除元素的大小，更大则向下调整，更小则向上调整。为了简化代码，去除判断，直接down和up操作都写上，只会执行一个

**5.修改任意一个元素**

直接修改，并且判断修改前后的新值与旧值大小关系，决定做down还是up。同删除任意元素操作，为了简化代码，去除判断，直接down和up操作都写上，只会执行一个。

**6.给定一个序列，构造小顶堆**：采用不断向下调整的方法构造。叶子节点无需再向下调整，只有非叶子结点需要向下调整，而**最后一个非叶子结点编号为$\frac{n}{2}$(从1开始编号)**，若从0开始编号为$\frac{n}{2}-1$([堆排序（完全二叉树）最后一个非叶子节点的序号是n/2-1的原因_完全二叉树最后一个非叶子结点的序号-CSDN博客](https://baolei.blog.csdn.net/article/details/105591415?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-105591415-blog-128136426.235^v43^control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-105591415-blog-128136426.235^v43^control&utm_relevant_index=2))

**down和up操作**

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1e6+10;
int h[N],lsize;//h[]存储结点，lszie为数组大小 从下标为1的位置开始存储

void down(int u){//传入的是数组下标
    int t=u;//承接u的值，进行遍历比较操作
    if(2*u<=lsize && h[2*u]<h[t]) t=2*u;//左子结点
    if((2*u+1)<=lsize && h[2*u+1]<h[t]) t=2*u+1;//右子结点
    if(t!=u){
        swap(h[u],h(t));
        down(t);//继续递归向下调整
    }
    
}

void up(int u){//这里也可以像down一样改成递归
    while(u/2 && h[u/2]>h[u]){
        swap(h[u/2],h[u]);
        u/=2;
    }
}


```

#### AcWing 838.堆排序

![img](assets\6b903107fe9844daa84c01c850113b8a.png)



```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1e6+10;
int h[N],lsize;//h[]存储结点，lszie为数组大小 从下标为1的位置开始存储

void down(int u){//传入的是数组下标
    int t=u;//承接u的值，进行遍历比较操作
    if(2*u<=lsize && h[2*u]<h[t]) t=2*u;//左子结点
    if((2*u+1)<=lsize && h[2*u+1]<h[t]) t=2*u+1;//右子结点
    if(t!=u){
        swap(h[u],h(t));
        down(t);//继续递归向下调整
    }
    
}

int mian(){
    int n,m;
    cin>>n>>m;
    lsize=n;
    for(int i=1;i<=n;i++) cin>>h[i];
    //构造堆
    for(int i=n/2;i;i--) down(i);
    while(m--){
        cout<<h[1]<<" ";//最小的数即为堆顶元素
        //调整堆
        h[1]=h[lsize];
        lsize--;
        down(1);
    }
    
}
```



#### AcWing 839.模拟堆

![img](assets\2b6a3e8af4174879a80b9d7298b27302.png)

**实现思路：**除了基本的down和up操作，这里还要求**删除和修改第k个数**，因此需要数组来记录数的下标和插入次序的关系

- 数组`ph[k]=i`：表示第k个插入的数的数组下标
- 数组`hp[i]=k`：表示数组下标为i的数是第几次插入的

**每次交换不再是单纯的交换值，而是需要交换对应关系，即ph[]和hp[]数组的映射关系也要修改**

```c++
#include <iostream>
#include <algorithm>
#include <string.h>
using namespace std;
const int N=1e6+10;
int h[N],lsize;
int hp[N],ph[N];//下标和插入次序的关系

//重新编写交换操作 因为不再是单纯的值交换
void heap_swap(int a,int b){//依旧传入的是下标
    swap(ph[hp[a]],ph[hp[b]]);//交换下标
    swap(hp[a],hp[b]);//交换插入次序
    swap(h[a],h[b]);//交换值
}

void down(int u){
    int t=u;
    if(2*u<=lsize && h[2*u]<h[t]) t=2*u;//左子结点
    if((2*u+1)<=lsize && h[2*u+1]<h[t]) t=2*u+1;//右子结点
    if(t!=u){
        heap_swap(h[u],h(t));
        down(t);//继续递归向下调整
    }
    
}

void up(int u){//这里也可以像down一样改成递归
    while(u/2 && h[u/2]>h[u]){
        heap_swap(h[u/2],h[u]);
        u/=2;
    }
}

int main() {
    int n,m=0;//m记录插入次序 即第几次插入
    cin>>n;
    while(n--)
    {
        char op[10];
        cin>>op;
        if(!strcmp(op,"I"))//插入一个数
        {
            int x;
            cin>>x;
           	h[++lsize]=x;
            ph[m++]=lsize;
            hp[lsize]=m;
            up(lsize);
        }
        else if(!strcmp(op,"PM"))//输出最小值
        {
            cout<<h[1]<<endl;
        }
        else if(!strcmp(op,"DM"))//删除最小值
        {
            heap_swap(1,lsize);
            lsize--;
            down(1);
        }
        else if(!strcmp(op,"D"))//删除第k个数
        {
            int k;
            cin>>k;
           	k=ph[k];
            heap_swap(k,lsize);
            lsize--;
            down(k),up(k);//省去判断，只会执行一个
        }
        else//修改第k'个数
        {
            int k,x;
            cin>>k>>x;
            k=ph[k];
            h[k]=x;
            down(k),up(k);//省去判断，只会执行一个
        }
    }
    return 0;
}

```



## 7.Hash表

**哈希表的作用：把一个比较大的空间，通过一个函数映射到一个比较小的空间。**

一般做哈希运算时，取一个质数作为模，会使得冲突的概率降低

**哈希函数的构造：**

- **除留余数法**

  $$H(key)=key\mod{p}$$

- **直接定址法**

  $H(key)=key$或$H(key)=a*key+b$

- **数字分析法**

  选取**数码分布较为均匀的若干位**作为散列地址。如选取手机号码后几位

- **平方取中法**

  取**关键字的平方值的中间几位**作为散列地址。



**哈希表的冲突解决方法**：

- **拉链法**

- **开放寻址法**

#### AcWing 840.模拟散列表

![img](assets\0cc14f0ec65840429e37fdd25dc668b2.png)

#### **（1）拉链法**

创建一个数组`h`，插入一个值时通过哈希函数映射到数组对应位置，**每个位置维护一个链表**，映射到相同位置值加入当前链表中。

**数组`h[i]`**类似于链表的头指针，存储的是其下链表的第一个结点`x`的**数组下标**，而**不是结点的值`x`**，取值范围0~N，所以可以**让数组h的默认值为-1**，**以此判断该位置下是否为空**

- 插入操作：**采用头插法**，根据哈希函数计算哈希值，每次冲突的值，插入到链表的第一个位置
- 查询操作：根据哈希值找到对应头指针即对应链表，再对链表遍历判断。
- 删除操作：删除操作并不是真正的删除该元素，而是设置一个标记值，来表示该位置的值已被删除（用得少）

**哈希函数常用取余法**，在本题中找一个大于或等于N且最小的质数

```c++
//求大于或等于N且最小的质数(N>1)
int get_prime(int N){
    for(int i=N;;i++){
        bool flag=true;
        for(int j=2;j*j<=N;j++)//在判断一个数是否为质数时，只需要检查其是否能被小于或等于其平方根的数整除。这是因为，如果一个数 n 能被一个比它大的数整除，那么这个因数必然有一个对应的比它小的因数。
            if(i%j==0) {
                flag=false;
                break;
            }
        
        if(flag){
            return i;
        }
    }
}
```

**代码实现：**

```c++
#include <iostream>
#include <cstring>
using namespace std;
const int N=100003;//大于10^5的最小质数为
int h[N],e[N],ne[N],idx=0;

//拉链法：插入操作 采用头插法
void insert(int x){
    int k=(x%N+N)%N;//求哈希值，+N再%N使负数求余为正数
    e[idx]=x;
    ne[idx]=h[k];
    h[k]=idx++;
}

//查找操作
bool query(int x){
    int k=(x%N+N)%N;//使负数求余也为正数
    for(int i=h[k];i!=-1;i=ne[i]){//i=-1时结束循环，即空指针
        if(e[idx]==x) return true;
    }
	return false;
}

int main() {
    memset(h,-1, sizeof(h));//令数组h初始化值为-1
    int n;
    cin>>n;
    while(n--)
    {
        char op[2];
        int x;
        cin>>op>>x;
        if(op[0] == 'I') insert(x);
        else{
            if(query(x)) puts("Yes");
            else puts("No");
        }
    }
    return 0;
}


```



#### **（2）开放寻址法**

开放寻址法：当冲突发生时，使用某种探测算法（**得出一个偏移量**）在散列表中寻找下一个空的散列地址，只要散列表足够大，空的散列地址总能找到。探测法有**线性探测法、平方探测法、双散列法、伪随机序列**。这里直接使用**线性探测法**，即冲突则自增1

**数组`h`**存储的是具体**结点值`x`**，而`x`取值范围是$[-10^9,10^9]$，故应让数组`h`的默认值不在`x`的取值范围中，这样才好判断`h[k]`位置上是否为空（注意和拉链法区分）

- 查找和查询操作合为一个**find函数**：首先根据哈希函数计算的哈希值查找当前元素是否在初始映射位置。若该位置为空，则在这个位置插入该元素；若不为空且与该元素不等，则向后继续查找，直到找到该元素或者有空位置则插入该元素。最后返回该位置

```c++
#include <iostream>
#include <cstring>
using namespace std;
//将大范围数字映射到小范围数字（10^5）首先将x通过哈希函数映射到哈希表h[x]中。（ps:该方法的哈希表范围要扩大至正常的2到3倍） 
const int N= 200003,null=0x3f3f3f3f;//null超出10^9
int h[N];

int find(int x){
    int k=(x%N+N)%N;//求哈希值，依旧要使负数的余数为正
    while(h[k]!=null && h[k]!=x){
        k++;
        if(k==N) k=0;//哈希表已满
    }
    return k;
}


int main() {
    memset(h,0x3f, sizeof h);//memset按字节赋值，int4个字节，每个字节赋值0x3f，则h默认值就为0x3f3f3f3f 即null
    int n;
    cin>>n;
    while(n--)
    {
        char op[2];
        int x;
        cin>>op>>x;
        int k=find(x);
        if(op[0] == 'I') h[k]=x;
        else{
            if(h[k]!=null) puts("Yes");
            else puts("No");
        }
    }
    return 0;
}

```



#### （3）字符串哈希

可以求解任意的子串的哈希值！ 这是KMP也望而却步的！**可以通过字符串哈希值，快速判断两个字符串是否相等或者两个字符串中某个部分是否相等**。（用模式匹配需要至少O(n)，而字符串哈希只需要O(1)）

**1.字符串哈希值**：实际为字符串**前缀**哈希值，如有字符串 `s = ABCDE`，用数组`h[]`存储其各个前缀哈希值，则`h[1] = A` 的哈希值；`h[2] = AB `的哈希值；`h[3] = ABC` 的哈希值.....

**2.如何求解一个字符串的哈希值？**

将字符串看成一个**P进制**的数，比如字符串ABCD，假如我们把A映射为1，B映射为2，C映射为3，D映射为4（实际上字母也直接取它的ASCII值也可）。将ABCD这个**P进制的数转化为十进制**即为其哈希值，则ABCD这个字符串的哈希值为：$(1 × P^3 + 2 × P^2+ 3 × P^1 + 4 × P^0  )\ mod \ Q$

> 最后modQ即防止转化的十进制数过大，用Q来缩小数据范围，就是哈希散列的意义
>
> **注意：**通常不要把一个字母映射为0，这样会导致重复。比如把A映射为0，则A是0，AA也是0，AAA还是0。
>
> 对于P、Q的取值，有一个经验值，将冲突概率降到极低。我们可以取 **P = 131或13331，Q = 2^64** 。为简化mod Q运算，可以将h数组的类型取成**unsigned long long（64位）**，这样就无需对2^64 取模，溢出就直接相当于取模

**可得一个递推公式，求解某个字符串`s[]`的（前缀）哈希值：即`h[i]=h[i-1]*P+s[i]`（类似求前缀和，只是这里每次要乘一个P）**注意：`i`从`1`开始；字符串是顺序存储在数组中，低位字符的权值大

**3.求解一个字符串某区间`[l,r]`上的字符串哈希值**

这就和前缀和有所区别，不是单纯的`h[r]-h[l-1]`。因为字符串是顺序存储在数组中，低位字符的权值大，区间`1~l-1`的字符串在相减过程中后面应该补0，**即抬高各字符的权值**，达到与`1~r`区间的字符串位数相等，有同等地位。如`1~l-1`是“ABC”，而`1~r`是“ABCDE”，求DE的哈希值，就应该让“哈希值（ABCDE）-哈希值（ABC00）”

**即体现到公式上，先算出两者相差的位数：`r-l+1`，然后`h[l-1]*P^(r-l+1)`，再相减得`[l,r]`区间字符串的哈希值=`h[r]-h[l-1]*P^(r-l+1)`**

注：为了简便的得到`P^i`的值，用一个数组来存储，即`p[i]=P^i`

**4.由以上步骤，即可通过哈希值来判断任意两个字符串是否相等**

#### AcWing 841.字符串哈希

![img](assets\4321a022ad644f34930debddfd77b74f.png)

```c++
#include <iostream>
#include <string>
using namespace std;
typedef unsigned Long Long ULL;//无符号的Long long数(64位)，不用再额外mod Q  Q = 2^64
const int N=1e5+10,p=131;//131或13331 
int n,m;
int h[N],P[N];//h：哈希值。P：进制权值
char str[N];

//得到某个区间字符串的哈希值
ULL get(int l,int r){
    return h[r]-h[l-1]*P[r-l+1];
}

int main(){
    cin>>n>>m>>str+1;//字符数组从1开始存储
    P[0]=1;
    for(int i=1;i<=n;i++){
        //计算P
        P[i]=P[i-1]*p;
        //计算前缀哈希值
        h[i]=h[i-1]*p+str[i];//这里字符直接用其ASCII值
    }
    while(m--){
        int l1,l2,r1,r2;
        cin>>l1>>r1>>l2>>r2;
        if(get(l1,r1)==get(l2,r2)) puts("Yes");
        else puts("No");
    }
    
    return 0;
}

```



## 8.C++中的**STL**(标准模板库)

**（1）vector**, 变长数组，倍增的思想  

- 头文件 `#include <vector>`

- ​    size()  返回元素个数
- ​    empty()  返回是否为空
- ​    clear()  清空
- ​    front()/back() 返回首、尾元素
- ​    push_back()/pop_back() 压入、弹出末尾元素
- ​    begin()/end() begin()=a[0]，end()=a[size]
- 支持比较运算，如a(3,4)，b(3,5)，a<b，即按字典序



**（2）pair<int, string>** 可以存储一个二元组

- ​    first, 第一个元素
- ​    second, 第二个元素
- ​    支持比较运算，以first为第一关键字，以second为第二关键字，按字典序

**（3）string，**字符串

- `#include <string>`

- ​    size()/length()  返回字符串长度
- ​    empty() 判断为空
- ​    clear() 清空
- ​    substr(起始下标，(子串长度))  返回子串
- ​    c_str()  返回字符串所在字符数组的起始地址

**（4）queue**, 队列

- `#include <queue>`

- ​    size()
- ​    empty()
- ​    push()  向队尾插入一个元素
- ​    front()  返回队头元素
- ​    back()  返回队尾元素
- ​    pop()  弹出队头元素

（5）**priority_queue,** 优先队列，即堆，默认是大根堆

- `#include <queue>`

- ​    size()
- ​    empty()
- ​    push()  插入一个元素
- ​    top()  返回堆顶元素
- ​    pop()  弹出堆顶元素
- ​    定义成小根堆的方式：priority_queue<int, vector<int>, greater<int>> q;

**（6）stack**, 栈

- `#include <stack>`

- ​    size()
- ​    empty()
- ​    push()  向栈顶插入一个元素
- ​    top()  返回栈顶元素
- ​    pop()  弹出栈顶元素

**（7）deque**, 双端队列

- `#include <deque>`

- ​    size()
- ​    empty()
- ​    clear()
- ​    front()/back()
- ​    push_back()/pop_back()
- ​    push_front()/pop_front()
- ​    begin()/end()

**（8）set, map, multiset, multimap**, 基于平衡二叉树（红黑树），动态维护有序序列
    size()
    empty()
    clear()
    begin()/end()
    ++, -- 返回前驱和后继，时间复杂度 O(logn)

**set/multiset**  set去重，mulset可以有重复元素

- ​    insert()  插入一个数
- ​    find()  查找一个数
- ​    count()  返回某一个数的个数
- ​    erase()
  - ​        (1) 输入是一个数x，删除所有x   O(k + logn)
  - ​        (2) 输入一个迭代器，删除这个迭代器
- ​    lower_bound()/upper_bound()
  - ​        lower_bound(x)  返回大于等于x的最小的数的迭代器
  - ​        upper_bound(x)  返回大于x的最小的数的迭代器

**map/multimap ** 

- ​    insert()  插入的数是一个pair
- ​    erase()  输入的参数是pair或者迭代器
- ​    find() 查找一个数
- ​    map可以像数组一样使用 ：如map<string,int> a; a["string"]=1；(注意multimap不支持此操作。 时间复杂度是 O(logn))
- ​    lower_bound()/upper_bound()

**（9）unordered_set, unordered_map, unordered_multiset, unordered_multimap,** 哈希表

- ​    和上面set, map, multiset, multimap类似，增删改查的时间复杂度是 O(1)
- ​    不支持 lower_bound()/upper_bound()， 迭代器的++，--

**（10）bitset,** 圧位，可以省8位空间，一个字节当8个字节用   bitset<10000> s;

- 支持各种位运算    ~, &, |, ^、`>>, <<`、==, !=
- count()  返回有多少个1

- any()  判断是否至少有一个1
- none()  判断是否全为0

- set()  把所有位置成1
- set(k, v)  将第k位变成v
- reset()  把所有位变成0
- flip()  等价于~
- flip(k) 把第k位取反



# 第三章 搜索与图论

## 1.DFS与BFS

DFS：深度优先搜索（Depth-First-Search）

BFS：宽度优先搜索（Breadth-First-Search）

#### **DFS和BFS的对比**

- DFS使用栈（stack）来实现，BFS使用队列（queue）来实现
- DFS所需要的空间是树的高度h，而BFS需要的空间是2^h （DFS的空间复杂度较低）

- DFS不具有最短路的特性，BFS具有最短路的特性

> BFS具有最短路的特性：当前每条边的权重相等时，BFS遍历从当前点到每个点的距离都是最短的

### （一）DFS

- 回溯：回溯的时候，一定要记得恢复现场
- 剪枝：提前判断某个分支一定不合法，直接剪掉该分支

经典问题：全排列问题、n皇后问题

#### **AcWing 842.排列数字**

![img](assets\1198e6c286be4c5a8c0560d12b3ad9e5.png)

**实现思路：**利用深度优先搜索

- 每次搜索确定一个位置的数字，然后回溯判断是否有其它可能，**回溯后要恢复现场**
- 设置一个结果数组`path[]`，存储每次确定放置的数字。dfs递归参数设置为`u`，表示要判断的第几个数字，若`u==n`表示找到一个序列输出，然后回溯（注意`u`从0开始，`u==n`时就意味着已经放置好了`n`个数）。
  - 设置一个bool数组`st[]`，`st[i]=true`表示数字i已经放置，否则未放置
  - 若未放置，则将`i`放置在当前位置`path[u]`，且设置对应数组`st[i]=true`
  - 然后递归到下一个为止`dfs(u+1)`，继续判断放置
  - 到达最后一个位置输出一次结果后，**回溯，恢复现场，设置`st[i]=false`**，然后继续循环判断这个位置放另一个数

```c++
#include <iostream>
using namespace std;
const int N=7;
int n;
int path[N],//path存储结果(从0开始)
bool st[N];//st表示当前数字是否已使用(放置)(从1开始) 初始化默认为false

//深度优先搜索
void dfs(int u){//u表示当前处理第几个数 总共三个 0 1 2... u=n时已处理n个
    if(u==n){//已找到一个结果
        for(int i=0;i<n;i++) cout<<path[i]<<" ";
        cout<<endl;
        return ;
    }
    //否则继续搜索
    for(int i=1;i<=n;i++){
        if(!st[i]){//当前数字未被使用
            path[u]=i;//使用
            st[i]=true;
            dfs(u+1);//递归
            st[i]=false;//回溯后要恢复现场
        }
    }
    
}

int mian(){
    cin>>n;
    dfs(0);//从0开始
    return 0;
}



```



#### **AcWing 843.n-皇后问题**

![img](assets\c73a552d6f3943cfb972ecbbade67aaa.png)

**方法一：按行枚举，dfs(u)**

- 确定n个位置，每次需要**回溯**和**剪枝**(即发现条件不符合时直接跳过这种情形，而不是把这种情况完成再进行排除，从而减少了时间复杂度操作。

- 条件判断包括**行、列、主对角线、副对角线**。u表示目前放置第几个皇后，也表示第几行。用列i作为循环条件，每次确定一行中某个位置`(u，i)`放置皇后

  - 对于行：由于每行只有一个皇后，要放置n个皇后，每次递归处理一行，放置一个皇后，当`u==n`时，结束表示放置完毕，即行为隐藏编号，无需额外的行数组

  - 对于列：设置一个列数组`col[]`，`bool`型，`col[i]`为true时表示当前列i已有皇后

  - 对于主对角线：同样设置一个`bool`数组`dg[]`，`dg[u+i]`为true时表示当前位置的主对角线方向有皇后

  - 对于副对角线：设置一个`bool`数组`udg[]`，`udg[i-u+n]`为true时表示当前位置的副对角线方向有皇后

    > 关于为什么是`u+i`，`i-u+n`：如果将棋盘类比成平面直角坐标系，左上角的点就是坐标原点O。可以把`u`看作横坐标，`i`看作纵坐标，若主对角线v1是不通过O的，那么v1上的点的横纵坐标之和不变，即`u+i`不变；副对角线v2上的点的横纵坐标之差不变即`|i-u|`绝对值不变，但是`i-u`会小于0（最小为0-8==-8），由于数组下标的限制，所以要对`i-u`加8。(如下图)
    > ![image-20240723155058400](assets\image-20240723155058400.png)

- 注意每次回溯前，即放置皇后后，要设置当前列、主对角、副对角为true；回溯后要设置当前列、主对角、副对角为false（即回到当前位置要恢复现场）

```c++
#include <iostream>
using namespace std;
const int N=20;
char g[N][N];//存储棋盘
bool col[N],dg[N],udg[N];
int n;

//深度优先搜索
void dfs(int u){//u表示行，从1开始
    if(u==n){//表示棋盘已经遍历完 皇后已经放好 输出一次结果
        for(int i=0;i<n;i++) puts(g[i]);//使用puts直接按行输出
        cout<<endl;
        return ;
    }
    //继续处理
    for(int i=0;i<n;i++){//按列循环处理
        if(!col[i] && !dg[u+i] && !udg[u-i+n]){//当前位置(u,i)可以放皇后
            g[u][i]='Q';//放置皇后
            col[i]=dg[u+i]=udg[u-i+n]=true;//更新标志
            dfs(u+1);//继续递归一行
            col[i]=dg[u+i]=udg[u-i+n]=false;//回溯后 恢复现场
            g[u][i]='.';
        }
    }
}

int mian(){
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            g[i][j]='.';//初始化棋盘
    dfs(0);//从0开始，表示第一行
    return 0;
}

```

**方法二：**将参数设置为更易于理解的三个参数，即**行号、列号、皇后放置数量**，**dfs(x,y,s)**

- 设置**四个数组，行数组row[]，列数组col[]，主对角数组dg[]，副对角数组udg[]**，分别表示当前位置的四个方位是否有皇后

- **从第一行第一列(0,0,0)开始**，每次首先判断当前行是否走完，走完就移动到下一行的第一列继续
  - 然后判断是否走到最后一行，若走到最后一行，则判断皇后是否放完，若放完就输出结果，结束
- 对于当前位置(x,y)有两种选择，不放皇后和放皇后；
  - **不放皇后**，直接递归到下一列dfs(x,y+1,s)；
  - **放皇后**。**进行判断当前位置(x,y)的四个方位是否符合条件**。若符合则放置皇后，更新四个数组为true，然后继续向下寻找下一个放置皇后的位置，然后回溯，恢复四个数组为false(恢复现场)

```c++
#include <iostream>
using namespace std;
const int N=20;
char g[N][N];
bool row[N],col[N],dg[N],dg[N];//增加一个row数组，记录当前行中是否已有皇后
int n;

//(x,y)坐标，s表示当前已放皇后数 均从0开始
void dfs(int x,int y,int s){
    if(y==n){x++;y=0;}//已经处理完一行 处理下一行
    if(x==n){//已处理完所有行
        if(s==n){//可能顺序遍历完后 皇后没有完全放好即s<=n
			//若皇后都放好了 输出结果
            for (int i = 0; i < n; i++) puts(g[i]);
            cout << endl;
        }
        return;
    }
    
    //行还未处理完 继续处理 两种情况
    //当前位置不放皇后
    dfs(x,y+1,s);//向下一列继续
    
    //当前位置放皇后 
    if(!row[x] && !col[y] && !dg[x+y] && !udg[x-y+n]){//若可放
        g[x][y]='Q';
        row[x]=col[y]=dg[x+y]=udg[x-y+n]=true;//更新标记
        dfs(x,y+1,s+1);//这里可不可以直接x+1?..
        row[x]=col[y]=dg[x+y]=udg[x-y+n]=false;//恢复现场
        g[x][y]='.';
    }
    
}

int main(){
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            g[i][j]='.';//初始化棋盘
    dfs(0,0,0);
    return 0;
}
```



### （二）BFS

**一般通过队列实现，当边的权值相同时具有最短性，可以求最少操作步数**.相比DFS无需回溯，而是逐层搜索。

#### **AcWing 844.走迷宫**

![img](assets\293558c0afde4ef78411fa9ea53776d5.png)

**实现思路：**广度优先搜索

- 设置一个二维数组`g[][]`存储图。设置一个队列`q[]`（`pair[int][int]`类型的，每个值存储坐标`(x,y)`），表示当前到达的坐标位置。设置一个距离数组`d[][]`，`d[x][y]`表示到原点的距离
- 当队列不为空时，取出队头元素，向四个方位进行判断，看是否可走，可走的路径更新距离，坐标再存入数组
  - 关于方位的选择，为简化代码，不用写四行判断来检验每个方位。设置两个一维数组`dx[4]`，`dy[4]`分别存储`{0,1,0,-1}`、`{1,0,,-1,0}`，然后循环四次，每次坐标加上对应的`dx`、`dy`，判断当前位置是否可走，若可走就更新坐标到原点的距离`d[x][y]`+1，并将新坐标加入队列

```c++
#include <iostream>
#include <cstring>
using namespace std;
const int N=10;
typedef pair<int><int> II;//用来表示队列存储的元素 即坐标(x,y)
int g[N][N],d[N][N];//图 距离数组表示(x,y)到原点的距离
II q[N*N];//队列  这里用数组模拟队列
int n,m;

//返回最终结果
int bfs(){
    memset(d,-1,sizeof d);//初始化距离数组为-1 表示当前坐标还未走过
    d[0][0]=0;
    q[0]={0,0};//队列初始化，从原点(0,0)开始
    int hh=0，tt=0;//队头、队尾指针
    int dx[4]={0,1,0,-1},dy={1,0,-1,0};//用来形成四个方位 简化代码
    while(hh<=tt){//队列不为空时
        auto t=q[hh++];//取出队头元素
        for(int i=0;i<4;i++){//四个方位循环判断哪条可走
            int x=t.first+dx[i],y=t.second+dy[i];//得到新坐标
            if(x>=0 && y>=0 && x<n && y<m && g[x][y]==0 && d[x][y]==-1){//这个坐标可以走
                d[x][y]=d[t.first][t.second]+1;//更新距离+1
                q[tt++]={x,y};//当前坐标入队
            }
        }
    }
    return d[n-1][m-1];//返回终点到原点的距离 即为结果
}

int main(){
    cin>>n>>m;
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            cin>>g[i][j];
    cout<<bfs[];
    return 0;
}
```

**补充：存储经过路径**

设置一个二维数组`p[][]`(`pair<int><int>`型)，`p[x][y]`存储当前位置坐标的前驱坐标，判断好要走的方位时，更新

然后从(n-1,m-1)开始遍历输出，即为从终点到原点经过坐标（也就是逆序输出）

```c++
II p[N][N];
p[0][0]={-1,-1};
if(x>=0 && y>=0 && x<n && y<m && g[x][y]==0 && d[x][y]==-1){
                d[x][y]=d[t.first][t.second]+1;
                q[tt++]={x,y};
    			p[x][y]=t;//记录当前坐标的前驱坐标 即从哪个点过来的
            }
//输出
int x=n-1,y=m-1;//终点到原点的路径输出
while(x>=0 && y>=0){
    cout<<x<<","<<y<<endl;
    auto t=p[x][y];//得到前驱
    x=t.first,y=t.second;
}
```



#### AcWing 845. 八数码

![img](assets\c131b518e07644f68ab44ab003678220.png)

**实现思路：**

- 首先题意：

![](assets\31041_b4125f44b)

- 可以**转化为求图的最短距离**，且权值为1，使用**BFS**
- **怎么表示状态（即每次移动后的矩阵是怎样的）？**显然用邻接矩阵、邻接表都不好
  - 将3x3的矩阵**按顺序转化为字符串**，即`s="123x46758"`
  - BFS要使用队列，这时队列就存储字符串`queue<string>`
- **如何记录到达每一个状态时已经移动的距离？**
  - 使用一个字典，字符串状态与已经移动的距离相对应，即`unordered_map<string,int>`
- 然后按照BFS的步骤进行。

**注：**

- 字符串下标`k`---->矩阵位置`(x,y)`：`x=k/3,y=k%3`
- 矩阵位置`(x,y)`---->字符串下标`k`：`k=x*3+y`

其他理解：

[AcWing 845. 八数码 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/940655/)

```c++
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_map>
using namespace std;

//找到字符串转移到目标状态的最少交次数
int bfs(string start){
    //定义目标状态
    string end="12345678x";
    //定义字符串队列  
    queue<string> q;
    q.push(start);//初始状态入队
    //定义距离数组
    unordered_map<string,int> d;//用SLT中的unordered_map
    d[start]=0;//初始状态距离为0
    
    //四个方向的转移
    int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
    
    while(q.size()){//只要队列不为空
        auto t=q.front();//得到队头元素
        int dist=d[t];//得到当前已转移的距离
        
        if(t==end) return dist;//如果现在已经转移到最终状态 退出
        
        //得到当前状态中x的位置 再转化为对应矩阵中的位置
        int k=t.find('x');//使用string中自带的find函数
        int x=k/3,y=k%3;
        
        //遍历四个可移动的方向
        for(int i=0;i<4;i++){
            //转移后的坐标
            int a=x+dx[i],b=y+dy[i];
            if(a>=0 && a<3 && b>=0 && b<3){//可以转移过去 没有越界
                swap(t[k],t[a*3+b]);//将x和可转移位置的字符交换位置
                //如果当前状态是第一次转移 记录到队列和距离数组
                if(!d.count(t)){
                    d[t]=dist+1;
                    q.push(t);
                }
                //还原状态 为另一个可转移方向做准备
                swap(t[k],t[a*3+b]);
            }
        }
        
    }
    //始终达不到目标状态 返回-1
    return -1;
}

int main(){
    string start;
    for(int i=0;i<9;i++){
        char c;
        cin>>c;
        start+=c;
    }
    cout<<bfs(start);
    return 0;
}
```



## 2.树与图的遍历、拓扑排序

> #### 树和图的存储
>
> 首先，**树是一种特殊的图（无环连通图）**。所以，这里只说图的存储即可。
>
> 首先，**图分为2种，有向图和无向图**。
>
> **有向图**中2个点之间的边是有方向的，比如`a -> b`，则只能从a点走到b点，无法从b点走到a点。
>
> **无向图**中2个点之间的边是没有方向的，比如`a - b`，则可以从a走到b，也可以从b走到a
>
> 通常，我们可以将无向图看成有向图。比如上面，对a到b之间的边，我们可以建立两条边，分别是a到b的，和b到a的。
>
> 所以，我们只需要考虑，有向图如何存储，即可。通常有2种存储方式:
>
> - **邻接矩阵**
>
>   用一个二维数组来存，比如g[a,b]存储的就是a到b的边。邻接矩阵无法存储重复边，比如a到b之间有2条边，则存不了。（用的较少，因为这种方式比较浪费空间，对于有n个点的图，需要n2的空间，这种存储方式适合存储稠密图）
>
> - **邻接表**
>
>   使用单链表来存。对于有n个点的图，我们开n个单链表，每个节点一个单链表。单链表上存的是该节点的邻接点（用的较多）

### (一)深度优先遍历DFS

**实现有向图，再通过设置两点之间相互指向就实现了无向图，而树是一种特殊的图，即无环连通图，因此实现有向图就可以解决绝大多数问题，一般有邻接矩阵和邻接表两种实现方式，常用邻接表（与哈希类似）**

#### **AcWing 846.树的重心**

![img](assets\5791b780f30e45dd8e72677ae09c1f4a.png)

<img src="assets\71bd0349ee947a69a6bec98b1e7272e9.png" alt="71bd0349ee947a69a6bec98b1e7272e9" style="zoom:50%;" />

题意：如上图，删除结点1后，剩下三个连通块，最大连通块中点的数量为4；删除结点2后最大连通块点数为6；删除4后最大连通块点数为5.....以此类推，最后得到这些数当中的最小值，该结点即为重心

**实现思路：**用深度优先遍历

- **图采用邻接表存储**，设置数组`h[]`，`h[i]`存储结点`i`的第一个相邻点(子结点)的下标，初始值为-1。以此构成每个`h[i]`都指向一条链表，采用头插法实现插入结点
- **图的深度优先遍历**：设置一个数组`st[i]`，表示结点`i`是否被访问过，每个结点仅被访问一次。访问每个结点，然后访问这个结点所连结的未访问的点，再以这个点为对象递归向下访问其连接的未访问的点（以访问链表的形式进行访问）
- **如何求某个结点删除后的最大连通块中的点数？以当前结点，分向下、向上两部分来计算比较**
  - 求当前结点**向下**的最大连通块中结点数：遍历访问当前结点的每个子树，记录该结点的子树中的具有最多的结点数`res`。同时记录当前结点及其所有子树的结点数`sum`（初始化为1即包含当前结点）
  - 求当前节点**向上**的连通块中结点数：即总结点数`n-sum`；
  - 然后两者中取大值：`res=max(res,n-sum)`，即为删除当前结点后最大连通块的结点数
- **最后比较各删除结点的最大连通块的结点数，取最小值**：`ans=min(ans,res)`，即为重心删除后的最大连通块的结点数

```c++
#include <iostream>
#include <cstring>
#inclulde <algorithm>
using namespace std;
const int N=1e5+10;
bool st[N];//判断当前结点是否已被访问
int h[N],e[N],ne[N],idx;//实现链接表 和单链表中的含义一样
int n;
int ans=N;//存储最终结果 因为是选出最小 所以初始化为最大的数

//插入结点 构造邻接表 头插法
void add(int a,int b){//a与b之间建立一条边 
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
    
}

//返回当前结点及其子树的结点总数
int dfs(int u){
    st[u]=true;//先标记一下当前结点被访问
    int sum=1,res=0;//sum表示当前结点及其子树的结点总数 初始化为1即先算一个当前结点 res表示删除该节点后的最大连通块的结点数
    for(int i=h[u];i!=-1;i=ne[i]){//循环遍历当前结点下的链表
        int j=e[i];//由链表获得子结点的值
        if(!st[j]){//如果这个子节点没有被访问过
            int s=dfs(j);//向下 获得结点及其子树的结点总数
            res=max(res,s);//得到当前结点下面的最大连通块结点数
            sum+=s;
        }
    }
    res=max(res,n-sum);//与上面的最大连通块比较取大者 
    ans=min(ans,res);
    return sum;
}

int mian(){
    cin>>n;
    memset(h,-1,sizeof h);//初始化为-1
    for(int i=0;i<n-1;i++){//构建n-1条边
        int a,b;
        cin>>a>>b;
        add(a,b);add(b,a);//无向图 添加两条边
    }
    dfs(1);//随便从一个点开始搜 这里从1开始
    cout<<ans;
    return 0;
}

```

### (二)宽度优先遍历BFS

#### **AcWing 847.图中点的层次**

![img](assets\c7da2e7096524f5a94000fc14244922b.png)

**实现思路：**权值都相等为1，可以使用广度优先遍历，具有最短路的特性

- 依旧使用邻接表的方式存储图
- 设置一个队列`q[]`，初始1号结点入队，队列不为空的时候就持续操作，每次处理完一个结点，就入队
- 设置一个数组`d[]`，表示当前结点与1号结点的距离，初始化为-1，等于-1时表示当前结点还未处理(也就同时充当了访问位true or false)，每次处理完一个结点就+1

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=1e5+10;
int q[N],d[N];
int h[N],e[N],ne[N],idx;//构造链表
int n,m;

//a和b之间添加边
void add(int a,int b){
    e[idx]=b;
    h[idx]=h[a];
    h[a]=idx++;
}

//返回 1号结点到n号结点的最短路距离
int bfs(){
    memset(d,-1,sizeof d);//距离初始化为-1
    int hh=0,tt=0;//依旧用数组模拟队列
    q[0]=1;//1号节点入队 下标从0开始
    d[1]=0;//1号结点到自身的距离为0
    while(hh<=tt){//队列不为空
        int t=q[hh++];//获得队头元素
        for(int i=h[t];i!=-1;i=ne[i]){
            int j=e[i];
            if(d[j]==-1){//表示该结点还未被访问
                d[j]=d[t]+1;//更新距离
                q[++tt]=j;//当前结点入队
            }
        }
    }
    return d[n];
}

int mian(){
    cin>>n>>m;
    memset(h,-1,sizeof h);//初始化为-1
    while(m--)
    {
        int a,b;
        cin>>a>>b;
        add(a,b);//有向图
    }
    cout<<bfs();
    return 0;
}

```

### (三) 宽度优先遍历的应用--拓扑序列

图的宽度优先搜索的应用，求拓扑序（拓扑序是针对**有向图**的）

1. 什么是**拓扑序**：将一个图的很多节点，排成一个序列，使得图中的所有边，都是从前面的节点，指向后面的节点。则这样一个节点的排序，称为一个拓扑序。
2. 若**图中有环，则一定不存在拓扑序**。
3. 可以证明，**一个有向无环图，至少存在一个入度为0的点，即一定存在一个拓扑序列**。有向无环图，又被称为拓扑图。

对于每个节点，存在2个属性，入度和出度。

- 入度，即，有多少条边指向自己这个节点。
- 出度，即，有多少条边从自己这个节点指出去。

每次以入度为0的点为突破口进行处理，每次处理得到一个新的入度为0的点，继续处理

#### **AcWing 848.有向图的拓扑序列**

![img](assets\27c0653db27249a9a8683d02d4820006.png)

**实现思路：**采用广度优先遍历

- 依旧使用邻接表存储图
- 设置一个数组`d[]`，表示当前结点的入度，每次添加边就要更新
- 设置一个队列，将入度为0的点入队，初始遍历所有结点将入度为0的点加入队列
- 然后队列不为空的时候取出队头元素，枚举队头元素的所有出边，然后删除，则指向结点的入度要减1，假如指向的节点入度刚好减为0了，则入队。然后继续处理

```c++
#include <iostream>
#include <cstring>
using namespace std;
const int N=1e5+10;
int q[N],d[N],h[N],e[N],ne[N],idx;
int n,m;

//ab添加边 构建邻接表
void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}

//拓扑排序 返回是否存在这个序列
bool topsort(){
    int hh=0,tt=-1;
    for(int i=1;i<=n;i++)
        if(!d[i]) q[++tt]=i;//遍历入队所有入度为0的点
    while(hh<=tt){
        int t=q[hh++];//队头元素出队
        for(int i=h[t];i!=-1;i=ne[i]){
            int j=e[i];
            d[j]--;//入度-1
            if(!d[j]) q[++tt]=j;//入度为0就入队
        }
    }
    return tt==n-1;//若存在拓扑序列
}

int mian(){
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);
        d[b]++;//b的入度加1
    }
    if(topsort()){//存在拓扑序列 实际就是队列中的数据
        for(int i=0;i<n;i++) cout<<q[i]<<" ";
        puts("");  
    }else{//不存在输出-1
        cout<<-1;
    }
    
}
```



## 3.最短路径

- 单源最短路
- 多源汇最短路在最短路问题中，**源点** 也就是 **起点**，**汇点** 也就是 **终点**

![img](assets\68747470733a2f2f696d67323032332e636e626c6f67732e636f6d2f626c6f672f323738313235352f3230323330322f323738313235352d32303233303231323233323731313136382d313139303832323139352e706e67)

#### **单源最短路**

单源最短路，指的是求**一个点**，到**其他所有点**的最短距离。（起点是固定的，单一的）

**所有边权都是正数**

- **朴素Dijkstra** $O(n^2)$，基于贪心
- **堆优化Dijkstra**  $O(mlogn)$ 两者孰优孰劣，取决于图的疏密程度（取决于点数n，与边数m的大小关系）。当是稀疏图（n和m是同一级别）时，可能堆优化版的Dijkstra会好一些。当是稠密图时（m和n^2是同一级别），使用朴素Dijkstra会好一些。

**存在负权边**

- **Bellman-Ford** O(*mn*)，基于离散数学，若有负权回路，可能会出现负无穷，即最短路不一定存在
- **SPFA** 一般：O(*m*)，最坏：O(mn)，比Bellman-Ford用得更多，要求图中不含负环，同时也可以求解正权边的最短路，一些情况下可以代替Dijkstra算法

#### **多源汇最短路**

- **Floyd** O(*n*^3)，基于动态规划

**最短路问题的核心在于，把问题抽象成一个最短路问题，并建图。图论相关的问题，不侧重于算法原理，而侧重于对问题的抽象。**

### （一）Dijkstra

**要求的边的权值为正**

#### **朴素Dijkstra**

主要思想：用一个集合**s**来存放最短距离已经确定的点。找到1号点到n号点的最短距离，时间复杂度为O(n^2)

1. 初始化距离数组`dist`表示1号点到某点的距离,1号点到本身的距离为0：`dist[1] = 0, 其他dist[i] = 0x3f3f3f3f`无穷大
2. 循环n次：每次从距离已知的点中，选取一个不在**`s`**集合中，且**距离起点最短**的点**`t`**（这一步后续可以用小根堆来优化），把**`t`**加入到集合**`s`**中。
3. 然后遍历一下所有边，看加入点**`t`**后，**`t`**的出边中，是否可以通过以**`t`**为中间点缩小与起点的距离，即`dist[j]>dist[t]+g[t][j]`
4. 当所有点都被遍历完后，判断`dist[n]`是否还为无穷大，若是则意味着1和n不连通，否则返回dist[n]即为1号点到n号点的最短路距离

朴素Dijkstra对应**稠密图**，因此用**邻接矩阵**来存储

##### AcWing 849. Dijkstra求最短路 I

![img](assets\4fe4b3ea064bcae4d194462ade573a2b.png)

**稠密图（m和n^2是同一级别）**，**用朴素dijkstra**

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#define INF 0x3f3f3f3f
using namespace std;
const int N=510;
int g[N][N],dist[N];//dist为到起点的距离
bool s[N];//表示当前点还有加入最短距离集合中
int n,m;

//返回最短距离
int dijkstra(){
    memset(dist,0x3f,sizeof dist);//初始化距离数组
    dist[1]=0;//到本身的距离为0
    for(int i=1;i<=n;i++){
        int t=-1;
        //找到当前不在最短中且距离起点最近的点
        for(int j=1;j<=n;j++){
            if(!s[j] && (t==-1 || dist[t]>dist[j])){ //t=-1一个点都还没
                t=j;
            }
        }
        
        s[t]=true;//标记为加入最短路
        
        //遍历所有点更新到起点的距离
        for(int j=1;j<=n;j++){
            if(dist[j]>(dist[t]+g[t][j]))
                dist[j]=dist[t]+g[t][j];
        }
    }
    if(dist[n]==0x3f3f3f3f) return -1;//不存在1到n的最短路
    else return dist[n];
}


int mian(){
    cin>>n>>m;
    memset(g,0x3f,sizeof g);//初始化图各边距离为无穷大
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        g[a][b]=min(g[a][b],c);//因为图中含有重边和自环 所以每次输入都更新为更小值
    }
    cout<<dijkstra();
    return 0;
}

```



#### 堆优化Dijkstra

当图为稀疏图时（点数n和边数m是同一数量级时，如下题），使用堆优化的dijkstra，**将时间复杂度由O(n^2)降为O(mlogn)**

堆可以自己手写（用数组模拟），也可以使用现成的（**C++的STL提供了优先队列priority_queue，即可使用大根堆和根堆，但相比手写堆不提供任意元素修改**）

**AcWing 850. Dijkstra求最短路 II**

![img](assets\08ad266e553fc804461087d2eb293e78.png)

**实现思路：堆优化，构造小根堆得到当前未加入点中和起点距离最小的点**

- **稀疏图，采用邻接表**存储图，但额外添加一个权重数组`w[]`，代表边a和边b的权值
- `pair<int,int> PII`第一个int表示1号点到该点的距离，第二个int就是该点的编号，利用C++STL优先队列建立小根堆`priority_queue<PII,vector<PII>,greater<PII>> heap` ，存储当前未加入最短路的点的集合
- **当堆heap不为空时，取出堆顶元素，即当前未加入中距离起点最近的点**，根据该点的出边，更新以该点为中间点后**各点与起点的距离是否缩小**，若缩小就更新
  - 更新距离时，可能对一些距离已知的点进行更新（更新为更小的距离），按理说，应该修改堆中对应节点的距离值，但由于优先队列中的堆不支持直接修改某元素的值，则**可以直接插入一个新的节点（此时对于同一个节点，堆中有两份，即冗余存储，不会出现覆盖）**，但没有关系，在默认情况下，`std::pair` 的比较首先比较第一个元素，如果第一个元素相等，再比较第二个元素。因此，如果两个 `pair` 的第一个 `int` 值相同，将会比较第二个 `int` 值谁更小，因此**更新距离后堆顶依旧是目前未加入点中和起点距离最小的点。**

```c++
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>//优先队列头文件
using namespace std;
typedef pair<int,int> PII;//第一个int是到起点距离，第二个int是当前点的编号
const int N=1e5+10;
int h[N],e[N],ne[N],w[N],idx;//构造邻接表
int dist[N];//到起点的距离
int n,m;
int s[N];//判断当前点是否已加入最短路中

//在a和b中加入一条边，权值为c  构造邻接表
void add(int a,int b,int c){
    e[idx]=b;
    ne[idx]=h[a];
    w[idx]=c;//存储边的权值
    h[a]=idx++;
}

int dijkstra(){
    memset(dist,0x3f,sizeof dist);//初始化距离数组
    dist[1]=0;
    //用优先队列建立小根堆
    priority_queue<PII,vector<PII>,greater<PII>> heap;
    heap.push({0,1});//起点入堆
    while(heap.size()){//堆不为空时
        auto t=heap.top();//得到堆顶元素  即到起点距离最近的点
        heap.pop();//删除堆顶元素
        int ver=t.second,distance=t.fist;//得到编号和距离
        if(s[ver]) continue;//如果已加入最短路则退出 重新取堆顶元素
        s[ver]=true;//标记加入最短路
        
        //遍历当前结点所连的结点 更新距离
        for(int i=h[ver];i!=-1;i=ne[i]){
            int j=e[i];//获得编号
            if(dist[j]>dist[ver]+w[i]){
                dist[j]=dist[ver]+w[i];
                heap.push({dist[j],j});//加入堆
            }
        }
    }
    if(dist[n]==0x3f3f3f3f) return -1;
    else return dist[n];
}

int main(){
    cin>>n>>m;
    memset(h,-1,sizeof h);//初始化链表
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    cout<<dijkstra();
    return 0;
}
```



### （二）**Bellman-Ford** 

该算法适用于有**负权边**的情况，注意：如果有**负权环**的话，最短路就不一定存在了。时间复杂度O(mn)。该算法可以求出来图中是否存在负权回路，但求解负权回路，通常用SPFA算法，而不用Bellman-Ford算法，因为前者的时间复杂度更低。

Bellman-ford不要求用邻接表或者邻接矩阵存储边，为简化操作，可以**定义一个结构体**，存储a，b，w。表示存在一条边a点指向b点，权重为w。则遍历所有边时，只要遍历全部的结构体数组即可

**主要步骤：**

- **循环n次**

  循环的次数的含义：假设循环了k次，则表示，从起点，经过不超过k条边，走到某个点的最短距离

- 每次循环，遍历图中所有的边。对每条边(a, b, w)，（指的是从a点到b点，权重是w的一条边）**更新`d[b] = min(d[b], d[a] + w)`**。该操作称为松弛操作。

  该算法能够保证，在循环n次后，对所有的边`(a, b, w)`，都满足`d[b] <= d[a] + w`。这个不等式被称为三角不等式。

#### AcWing 853. 有边数限制的最短路

![img](assets\e8e11f0487ce3d19969eb07e1a932ea2.png)

**实现思路：**

- 利用上述的Bellman-ford算法

- 依旧定义一个距离数组`dist`，初始化为正无穷(0x3f3f3f3f)

  **注意**最后判断到n号结点是否有路径不是直接判断`dist[n]==0x3f3f3f3f`，**因为存在负权边，可能更新的时候会存在`dist[n]=0x3f3f3f3f-c`**，即无穷大加上一个负数，仍为无穷大，但数值还是改变了，**所以最后是否有路径的判断改为`dist[n]>0x3f3f3f3f/2`**

- 本题要求1号到n号点**不超过`k`条边**的最短距离，则循环k次来寻找最短路。

- 每次再遍历m条边，判断加入当前点后，各点到起点的距离是否变小，若变小则更新距离

  **注意**：该距离更新时可能会导致参与的边的数量大于k，因此应在每次遍历前设置一个**备份数组backup**，记录在k次遍历中，本次遍历的上一次的距离数组状态，在该次遍历中对每条边的距离数组更新时采用备份数组，确保本次更新范围在当前的边数限制内。

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=510,M=10010;
int dist[N],backup[N];//距离数组和备份数组
int n,m,k;
//定义一个结构数组存储边 
struct Edge{
    int a,b,w;
}edge[M];

int bellman_ford(){
    memset(dist,0x3f ,sizeof dist);
    dist[1]=0;
    for(int i=0;i<k;i++){//限定了最多k条边
        memcpy(backup,dist,sizeof dist);//先将距离数组做一次备份
        //遍历m条边 更新距离
        for(int j=0;j<m;j++){
            int a=edge[j].a,b=edge[j].b,w=edge[j].w;
            if(dist[b]>(backup[a]+w))//用备份数组更新
                dist[b]=backup[a]+w;
        }
    }
    if(dist[n]>0x3f3f3f3f/2) return -1;
    else return dist[n];
}

int main(){
    cin>>n>>m>>k;
    for(int i=0;i<m;i++){//下标从0开始
        int a,b,w;
        cin>>a>>b>>w;
        edge[i]={a,b,w};
    }
    int t=bellman_ford();
    if(t==-1) cout>>"impossible";
    else cout>>t;
    return 0;
}


```



### （三）SPFA

若要使用SPFA算法，一定要求**图中不能有负权回路**。**只要图中没有负权回路，都可以用SPFA**，即也可以求解正权边的题，这个算法的限制是比较小的。**时间复杂度为一般为O(m)，最差为O(mn)，在一些情况下可以代替Dijkstra算法**

**SPFA其实是对Bellman-Ford的一种优化，相比Bellman-Ford判环的时间复杂度也更低。**

它优化的是这一步：`d[b] = min(d[b], d[a] + w)`

我们观察可以发现，**只有当`d[a]`变小了，在下一轮循环中以a为起点的点（或者说a的出边）就会更新，即下一轮循环必定更新`d[b]`**（只有我变小了，我后面的点才会变小）

考虑用一个队列queue，来存放距离变小的节点。（当图中存在负权回路时，队列永远都不会为空，因为总是会存在某个点，在一次松弛操作后，距离变小）（和堆优化Dijkstra很像）

#### **AcWing** 851. spfa求最短路

![img](assets\8a4b3050ddff53ba56992d70e72f5838.png)

**实现思路：**相比上题没有k条边的限制，所以使用SPFA算法，时间复杂度更低

- **稀疏图，使用邻接表存储**，类似dijkstra

- 设置一个**队列`queue`**（使用C++STL中提供的队列queue）来存储待更新的点，初始将起点入队，**同时设置一个数组`s[]`作为标记，标记该点在队列中**，然后在每次遍历中，弹出队头，同时将该点取消标记，循环判断该点的相连的节点距离是否变小，**若变小，则更新距离**。**同时不在队列中，则入队同时标记为已入队**

  因为遍历过程中可能遍历到的点已经在队列中，但依旧可以更新距离，所以这里的判断不是同时判断距离变小和是否在队列中，要分开判断

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;
const int N=1e5+10;
int e[N],ne[N],w[N,h[N],idx;
int dist[N];
int n,m;
bool s[N];//记录是否在队列中

void add(int a,int b,int w){
    e[idx]=b;
    ne[idx]=h[a];
    w[idx]=w;
    h[a]=idx++;
}                 

int spfa(){
    memset(dist,0x3f,sizeof dist);
    dist[1]==0;
    queue<int> q;//建立一个队列
    q.push(1);//起点入队
    s[1]=true;//入队标记
    while(q.size()){//队列不为空
        auto t=q.front();//获得队头元素
        q.pop();//队头出队
        s[t]=false;//出队 更改标记
        
        //遍历 更新距离
        for(int i=h[t];i!=-1;i=ne[i]){
            int j=e[i];
            if(dist[j]>dist[t]+w[i]){//距离变小
                dist[j]=dist[t]+w[i];
                if(!s[j]){
                    s[j]=true;
                    q.push(j);//入队
                }
            }
        }
    }
    if(dist[n]>0x3f3f3f3f/2) return -1;
    else return dist[n];
}                 
                 
                 
int mian(){
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while(m--){
        int a,b,w;
        cin>>a>>b>>w;
        add(a,b,w);
    }
    int t=spfa();
    if(t==-1) cout<<"impossible";
    else cout<<t;
    return 0;
}                 
```



#### **AcWing** 852. spfa判断负环

![img](assets\caf2bc842293a73073e4aa23057ddf15.png)

**实现思路：**

- 在以上SPFA的基础上，设置一个**记录边数的数组`cnt[]`**，即代表起点到当前点经过的边数

- **在每次更新某点到起点最小距离的同时，将该点的cnt数组值加1**，即经过的边数多了一条，当该数组的值大于n，则说明存在负环（n个点若不存在环，最多只有n-1条边）。
- 注意：**由于判断的不是从起点开始存在负环，而是全部图中是否存在负环，因此初始要将所有点都放入队列当中。**

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;
const int N=1e5+10;
int e[N],ne[N],w[N,h[N],idx;
int dist[N],cnt[N];//cnt数组记录起点到当前点的边数
int n,m;
bool s[N];//记录是否在队列中

void add(int a,int b,int w){
    e[idx]=b;
    ne[idx]=h[a];
    w[idx]=w;
    h[a]=idx++;
}                 

bool spfa(){
    memset(dist,0x3f,sizeof dist);
    dist[1]==0;
    queue<int> q;//建立一个队列
    
    //所有点都要先入队
    for(int i=1;i<=n;i++){
        q.push(i);
    	s[i]=true;
    }
    
    while(q.size()){//队列不为空
        auto t=q.front();//获得队头元素
        q.pop();//队头出队
        s[t]=false;//出队 更改标记
        
        //遍历 更新距离
        for(int i=h[t];i!=-1;i=ne[i]){
            int j=e[i];
            if(dist[j]>dist[t]+w[i]){//距离变小
                dist[j]=dist[t]+w[i];
                cnt[j]++;//边数加1
                if(cnt[j]>=n) return true;//存在环
                if(!s[j]){
                    s[j]=true;
                    q.push(j);//入队
                }
            }
        }
    }
   return false;//不存在环
}                 
                 
                 
int mian(){
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while(m--){
        int a,b,w;
        cin>>a>>b>>w;
        add(a,b,w);
    }
    if(spfa()) cout<<"Yes";
    else cout<<"No";
    return 0;
} 
```



### （四）Floyd

求解**多源汇最短路问题(任意两点的最短距离)**，也能处理边权为负数的情况，但是**无法处理存在负权回路的情况。**

使用**邻接矩阵**来存储图

算法思路：三层循环，第一重循环为k，代表中间结点（顺序不可以改变）；后两层循环为i，j（顺序可调换），然后更新`d[i][j]=min(d[i][j],d[i][k]+d[k][j])`

#### **AcWing** 854. Floyd求最短路

![image-20240725224329952](assets\image-20240725224329952.png)

注意：本题可能存在重边，因此在读入时要选取更小的权值的边。初始化时，所有`d[i][i]=0`，其他为正无穷INF

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=210,INF=1e9;
int d[N][N];
int n,m,Q;

void floyd(){
    for(int k=1;k<=n;k++)
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                d[i][j]=min(d[i][j],d[i][k]+d[k][j]);
}

int mian(){
    cin>>n>>m>>Q;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++){
            if(i==j) d[i][j]=0;
            else d[i][j]=INF:
        }
    while(m--){
        int a,b,w;
        cin>>a>>b>>w;
        d[a][b]=min(d[a][b],w);//因为可能含有重边
    }
    floyd();
    while(Q--){
        int a,b;
        cin>>a>>b;
        if(d[a][b]>INF/2) cout<<"impossible"<<endl;
        else cout<<d[a][b];
    }
    return 0;
}


```



## 4.最小生成树

**最小生成树**：由n个节点，和n-1条边构成的无向连通图被称为G的一颗生成树，在G的所有生成树中，边的权值之和最小的生成树，被称为G的最小生成树。(换句话说就是用最小的代价把n个点都连起来)

- **Prim算法（普利姆）**
  - 朴素版Prim（时间复杂度**O(n^2)**，适用于**稠密图**）
  - 堆优化版Prim（时间复杂度O(mlogn)，适用于稀疏图）基本不用
- **Kruskal算法（克鲁斯卡尔）**适用于**稀疏图**，时间复杂度**O(mlogm)**

如果是**稠密图，通常选用朴素版Prim算法**，因为其思路比较简洁，代码比较短，如果是**稀疏图，通常选用Kruskal算法**，因为其思路比Prim简单清晰。堆优化版的Prim通常不怎么用。

### **（一）Prim**

#### 朴素Prim

与朴素dijkstra思想几乎一样，只不过Prim算法的距离指的是**点到最小生成树的集合的距离**，而dijkstra算法的距离是**点到起点的距离**。适用于稠密图

**实现思路：**和朴素Dijkstra很像

- 初始化**各点到集合的距离**为INF

- n次循环，每次找到**集合外且距离集合最近的点**`t`，需要先判断除第一个点外找到的距离最近的点`t`距离是不是INF

  - 若是则不存在最小生成树了，结束；否则可能还存在，继续操作，用该点`t`来**更新其他点到集合的距离**（这里就是和Dijkstra最主要的区别），然后将该点t加入集合

  **关于到集合的距离最近的点**：实际上就是不在集合中的点与集合内的点的各个距离中的最小值，每次加入新的点都会尝试更新一遍

  `dist[j]=min(dist[j]，g[t][j])`（在Dijkstra中是`dist[j]=min(dist[j]，dist[t]+g[t][j])`）

  ![请添加图片描述](assets\214fe860c631ef3433a37fc6d1e6fb59.jpeg)

##### **AcWing 858. Prim算法求最小生成树**

![img](assets\e4bc762f3b618a5418cc0292bab70375.png)

**注意：**本题由于未设起点所以要迭代n次，并且图中可能存在负的自环，因此**计算最小生成树的总距离要在更新各点到集合距离之前**。且该图为无向图，含重边，则构建边要注意

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=1e5+10,INF=0x3f3f3f3f;
int g[N][N],dist[N];
bool st[N];
int n,m;

//返回最小生成树权重之和
int prim(){
    memset(dist,0x3f,sizeof dist);
    int res=0;//记录结果
    
    for(int i=0;i<n;i++ ){//i从0开始 1开始无所谓
        int t=-1;
        
        //找到当前的集合外距离集合最近的点
        for(int j=1;j<=n;j++)
            if(!st[j] && (t==-1 || dist[t]>dist[j])) t=j;
        
        //判断该点不是第一个点 且到集合距离为INF
        if(i && dist[t]==INF) return INF; //不存在最小生成树
        if(i) res+=dist[t];
        st[t]=true;//标记进入集合
        //更新距离
        for(int j=1;j<=n;j++){
            dist[j]=min(dist[j],g[t][j]);
        }
        
    }
    return res;
}

int mian(){
    cin>>n>>m;
    memset(g,0x3f,sizeof g);
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        g[a][b]=g[b][a]=min(g[a][b],c);
    }
    int t=prim();

    if(t==INF) puts("impossible");
    else cout<<t;
    return 0;

}

```



**堆优化Prim**

思路和堆优化的Dijkstra思路基本一样，且基本不用，对于稀疏图，不如用Kruskal，这里略过



### （二）Kruskal

适用于**稀疏图**，时间复杂度**O(mlogm)**

1. 先将所有边按照权重，从小到大排序（快排，使用sort函数） O(mlogn)
2. 从小到大枚举每条边(a，b，w)，若a，b不连通，则将这条边，加入集合中（将a点和b点连接起来） **实质上并查集的一个应用（两点之间加边、看两点是否在一个连通块）**，时间复杂度为O(m)

​	无需邻接表/邻接矩阵存储图，直接使用结构体，表示边及其权值

<img src="assets\884b6c011f422611fcf875f437f53107.png" alt="在这里插入图片描述" style="zoom:67%;" />

#### **AcWing** 859. Kruskal算法求最小生成树

![img](assets\348389ad99b9efb0f568b1ac5189863b.png)

**实现思路**：借助并查集

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=200010;
int p[N];//父节点数组
int n,m;
struct Edge{
    int a,b,w;
    //重载一下 使用sort函数时按照w的值排序
    bool operator <(const Edge &W)const{
        return w < W.w;
    }
}edge[N];

//并查集：找集合根结点
int find(int x){
    if(p[x]!=x) return p[x]=find(p[x]);
    return p[x];
}

int main(){
    cin>>n>>m;
    for(int i=0;i<m;i++){
        int a,b,w;
        cin>>a>>b>>w;
        edge[i]={a,b,w};
    }
    
    sort(edge,edge+m);//按照边的权值排序
    for(int i=1;i<=n;i++) p[i]=i;//并查集初始化各个结点
    
    int res=0,cnt=0;//res存储最小生成树权值之和 cnt生成树边数 判断是否存在最小生成树
    
    //遍历所有边
    for(int i=0;i<m;i++){
        int a=edge[i].a,b=edge[i].b,w=edge[i].w;
        //找到两点各自的根结点 用于判断两点是否连通
        a=find[a],b=find[b];
        if(a!=b){//两点不连通 要使其连通
            p[a]=b;//并查集的合并
            res+=w;//累计权值
            cnt++;//边数++
        }
    }
    if(cnt<n-1) puts("impossible");//无最小生成树
    else cout<<res;
    return 0;
}


```



## 5.二分图：染色法、匈牙利算法

**二分图**：可以将一个图中的所有点，分成左右两部分，使得图中的所有边，都是从左边集合中的点，连到右边集合中的点。而左右两个集合内部都没有边。

![img](assets\68747470733a2f2f706963342e7a68696d672e636f6d2f38302f76322d61366264303033323039326130343931386139336639303138376535336338622e706e67)

再通俗点理解：**一个图是二分图当且仅当图中不含奇数环（奇数环：边数为奇数的环）**

### （一）染色法--判断二分图

对每一个点进行**染色操作**，只用黑白两种颜色；用**dfs和bfs**两种方式去实现,对图进行遍历并染色。时间复杂度为**O(n+m)**

**二分图一定不含奇数环，不含奇数环的图一定为二分图**

**充分性：**如果图中存在奇数环（构成环的顶点数量是奇数），那一定不是二分图），下图可以看到，依次选一个点，进行染色（原则是相邻的点要染于该点不同色），奇数环的染色结果会出现矛盾。；

<img src="assets\2f0207e017a82c1b9ece3fab96f46e09.png" alt="在这里插入图片描述" style="zoom: 50%;" />

**必要性：**如果没有奇数环，那么剩下的点的关系就是：偶数环or单链。这两种情况都能保证同一条边上相邻顶点在不同集合中，所以也是成立的；

<img src="assets\eca413fa9c9b7fb1bab7a0ad4238cdde.png" alt="在这里插入图片描述" style="zoom:50%;" />

综上：只要在染色过程中不存在矛盾（这里用黑白进行染色，即一个点不能即为黑色，又为红色），整个图遍历完成之后，所有顶点都顺利染上色。就说明这是一个二分图

#### AcWing 860. 染色法判定二分图

![img](assets\0bc74f4595750aba084b769ba11057c6.png)

**实现思路：**这里使用深度优先遍历DFS实现（代码相比BFS短点）

- 使用邻接表存储图

- 使用一个数组`color`代表当前点的染色，若为`-1`则未染色，若为`0`则为白色，若为`1`则为黑色
- 设置一个标志变量`flag`判断是否矛盾。然后循环遍历各点，若未染色则进行染色（染为0或1），对该点再进行深度优先遍历，对其连通的点进行判断。
  - 若未染色，则染上与该点相反的颜色；若已染色，判断是否与当前点的颜色相同。
    - 若相同则出现矛盾，提前返回false，得到结果不是二分图。否则继续遍历判断、染色

- 直到循环结束，判断标志变量flag，为真则为二分图

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=1e5+10,M=2*N;
int h[N],e[M],ne[M],idx;
int color[N];//结点颜色
int n,m;

void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}

//深度优先遍历染色 若出现矛盾返回false
bool dfs(int u,int c){//u表示结点,c表示染色种类 未染-1 白0 黑1
    color[u]=c;//先染色
    //给相连点染色
    for(int i=h[u];i!=-1;i=ne[i]){
        int j=e[i];
        if(color[j]==-1){//未染色
            if(!dfs(j,!c)) return false;//染上相反色，且对其继续深度遍历染色其他结点直至返回false出现矛盾
        }eles if(color[j]==c) return false;//若已经染色 且和当前颜色相同 则矛盾
    }
    return true;//未出现矛盾
}

int main(){
    cin>>n>>m;
    memset(h,-1,sizeof h);
    memset(color,-1,sizeof color);//初始化一下染色数组
    
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b),add(b,a);//无向图
    }
    
    bool flag=true;//判断是否是二分图
    for(int i=1;i<=n;i++){
        if(color[i]==-1){//未染色
            if(!dfs(i,1)){//染色 然后深度遍历染色
                flag=false;//出现矛盾 不是二分图
                break;
            }
        }
    }
    if(flag) puts("Yes");
    else puts("No");
    return 0;
}

```





### （二）匈牙利法--二分图的最大匹配

**二分图的最大匹配：**

- **匹配**（本质是一个边的集合！）
  给定一个二分图S，在S的一个子图M中，M的边集{E}中的任意两条边都不依附于同一个顶点，则称M是一个匹配。

- **极大匹配**

  极大匹配是指在当前已完成的匹配下,无法再通过增加未完成匹配的边的方式来增加匹配的边数。（也就是说，再加入任意一条不在匹配集合中的边，该边肯定有一个顶点已经在集合中的边中了）

- **最大匹配**

  所有极大匹配当中边数最大的一个匹配

下图是一个最大匹配（黄色边）

<img src="assets\cab91ea75bfe23fcf0b663dc40a6b7b7.png" alt="在这里插入图片描述" style="zoom:50%;" />

通俗一点的理解：

![在这里插入图片描述](assets\e9bf178ba8f8b09a4d6f394b06cabb8f.png)

**匈牙利算法：**两个集合中都存在一些点，以左集合出发，查找两个集合之中匹配成功的点的个数。如果左集合的某一个点发现与自己相连的节点已经被占有，则查询占有该节点的左集合的点是否有其他可配对的点，若有则两全其美，否则继续寻找，若仍未找到，则配对失败。**时间复杂度理论上是O(nm)，但实际运行时间一般远小于O(nm)。**

通俗点说：从左边开始，左边a喜欢右边b（即两者有连线），若b此时没有匹配的人或者即便有匹配的人，那个匹配的对象可以找到其他人（下家）来配对，即把b让给a，则a与b匹配成功。若两种情况下都不满足，则a只能找下一个有好感的人尝试匹配，直至匹配成功或已经没有喜欢的了。进行下一个人的匹配，最终得到的匹配对数即为二分图的最大匹配对

> 更加详细的理解：[【二分图算法】手把手教你学会：染色法（判断二分图）、匈牙利算法（二分图的最大匹配）_二分图染色的原理-CSDN博客](https://blog.csdn.net/Yaoyao2024/article/details/129895964?ops_request_misc=%7B%22request%5Fid%22%3A%22172199890116800180630060%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=172199890116800180630060&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-129895964-null-null.142^v100^control&utm_term=二分图染色法&spm=1018.2226.3001.4187)



#### AcWing 861. 二分图的最大匹配

![img](assets\c1b24b475483a680c297a2529b8fd906.png)

**实现思路：**

- 使用邻接表存储图。以左边集合为主，每个点可能与多个点相连（存在多个可能的匹配）
- 设置一个**判重数组`s[]`**，`s[i]`为真表示`i`结点已经被当前结点尝试过，**每次换一个人都要把数组`s[]`重置为false**；设置一个**匹配数组`match[]`**，为0表示当前结点还没有匹配的对象，否则为匹配对象的编号
- **find函数（找匹配对象）**：每次对相连的点尝试匹配，若当前连接点未尝试过，则尝试；若该点没有匹配对象或者他的匹配对象可以把该结点让给我，则与之匹配，退出匹配成功，下一个继续

```c++
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
const int N=510,M=1e5+10;
int h[N],ne[M],e[M],idx;
int match[N];//匹配数组
bool s[N];//判重数组
int n1,n2,m;

void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}

//找到x的匹配对象
bool find(int x){
    //遍历相连的点
    for(int i=h[x];i!=-1;i=ne[i]){
        int j=e[i];
        if(!s[j]){//还没尝试过 则尝试
            s[j]=true;
            //如果没有对象或者她对象有下家可以把她让给你
            if(match[j]==0 || find(match[j])){
                match[j]=x;
                return true;//匹配成功
            }
        }
    }
    return false;//尝试下来 匹配失败
}


int mian(){
    cin>>n1>>n2>>m;
    memset(h,-1,sizeof h);
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);//虽然是无向图，但只从一边的集合开始遍历，不用反过来再存储一遍了
    }
    
    int cnt=0;//记录最大匹配对数
    for(int i=1;i<=n;i++){
        memset(s,false,sizeof s);//每次给一个点匹配时，先将s置false 表示都还没尝试
        if(find(i)) cnt++;//i找到对象了
    }
    cout<<cnt;
    return 0;
}
```



# 第四章 数学知识

## （一）数论

### 1.质数

1. 对所有的大于1的自然数字，定义了【质数/合数】这一概念。对于所有小于等于1的自然数，没有这个概念，它们既不是质数也不是合数。
2. **质数的定义**：对于大于1的自然数，如果这个数的约数（因数）只包含1和它本身，则这个数被称为质数，或者素数



#### （1）试除法判定质数

**试除法**：对于一个数n，从2枚举到n-1，若有数能够整除n，则说明除了1和n本身，n还有其他约数，则n不是质数；否则，n是质数

**优化**：由于一个数的约数都是成对出现的。比如12的一组约数是3，4，另一组约数是2，6。则我们只需要枚举较小的那一个约数即可

我们用d|n来表示**d整除n(n%d=0)**，**只要满足d|n，则一定有{n/d}|n**，比如3∣12，则{12/3} | 12，**因为约数总是成对出现的，我们只需要枚举小的那部分数即可**，令$d ≤ n/d$，即，$d ≤ \sqrt{n}$，因此对于n，只枚举2到$\sqrt{n}$即可。

注意：for循环的结束条件，推荐写成`i <= n / i`。有的人可能会写成`i <= sqrt(n)`，这样每次循环都会执行一次sqrt函数，而这个函数是有一定时间复杂度的。而有的人可能会写成`i * i <= n`，这样当`i`很大的时候（比如i比较接近int的最大值时），`i * i`可能会溢出，从而导致结果错误。

##### AcWing 866. 试除法判定质数

![img](assets\22dbc79a2749dde39d700ecc378e6ef0.png)

```c++
#include <iostream>
#include <algorithm>
using namespace std;

bool is_prime(int x){
    if(x<=1) return false;
    for(int i=2;i<=x/i;i++){
        if(x/i==0) return false;
    }
    return true;
}

int main()
{
    int n;
    cin>>n;
    while(n--)
    {
        int x;
        cin>>x;
        if(is_prime(x)) puts("Yes");
        else puts("No");
    }
    return 0;

```



#### （2）**质因数分解-试除法**

对于一个整数 N 总能写成如下形式：
$$
N=P_{1} ^{α_1} \times P_{2} ^{α_2} \times P_{3} ^{α_3} \dots \times P_{n} ^{α_n}
$$
其中$P_i$都是质数，$α_i$为大于0的正整数，即**一个整数可以表示为多个不同质数的次方的乘积**

**对于一个数求解质因数的过程**：**从2到n，枚举所有数，依次判断是否能够整除 n 即可（朴素法，时间复杂度O(n))**。

**优化**：n中只包含一个大于$\sqrt{n}$的质因子，很好证明，如果中包含两个大于$\sqrt{n}$的质因子，那么乘起来就大于n了。因此，在枚举的时候可以先把2到$\sqrt{n}$的**质因子枚举出来**，如果最后处理完后剩下的数>1，那么这个数就是那个大于$\sqrt{n}$的质因子，单独处理一下就可以。**时间复杂度降为O($\sqrt{n}$)**

> 求质因数分解，为什么枚举所有数，而不是枚举所有质数，万一枚举到合数怎么办？解释：枚举数时，对于每个能整除 **n** 的数 **i**，**先把这个数除干净了**（就是把这个质数的次方剔除了，**表现在上式中就是逐步去除$P_i^{α_i}$**），再继续枚举后面的数，这样能保证，后续再遇到能整除的数，一定是质数而不是合数。

例如：求180的质因数分解

1. i = 2， n = **180** / 2 = **90** / 2 = **45**
2. i = 3 ，n = **45** / 3 = **15** / 3 = **5**
3. i = 4 ，当`i`是合数时，`i` 一定不能整除 `n `。如果 4 能整除 `n` 那么 2 一定还能整除 `n`，就是在` i = 2`的时候没有除干净，而我们对于每个除数都是除干净的，因此产生矛盾。
4. i = 5 ，n = **5** / 5 = **1**

##### AcWing 867. 分解质因数

![img](assets\a9ff9ce54843d6069c7b5a9dab0af63d.png)



```c++
#include <iostream>
using namespace std;

void divide(int x){
    for(int i=2;i<=x/i;i++){
        if(x%i==0){//找到一个质因数
            int s=0;//记录质因数的指数
            while(x%i==0){//除干净
                x/=i;
                s++;
            }
            printf("%d %d\n",i,s);
        }
    }
    if(x>1) printf("%d %d\n",x,1);//除到最后x还大于1 那x也为质因数 指数为1
    puts(" ");
}


int main()
{
    int n;
    cin>>n;
    while(n--)
    {
        int x;
        cin>>x;
        divide(x);
    }
}
```



#### （3）筛质数--朴素法

将2到n全部数放在一个集合中，遍历2到n，**每次删除当前遍历的数在集合中的倍数**。最后集合中剩下的数就是质数。

解释：如果一个数p没有被删掉，那么说明在2到p-1之间的所有数，p都不是其倍数，即2到p-1之间，不存在p的约数。故p一定是质数。

> 时间复杂度：
> $$
> \frac{n}{2} + \frac{n}{3} + \dots +\frac{n}{n} = n\ln_{}{n} < n\log_{2}{n}
> $$
> 故，朴素思路筛选质数的时间复杂度大约为**O(nlogn)**

##### AcWing 868. 筛质数

![img](assets\e84b98263697e2698582ef51bead4eee.png)

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int primes[N];//存储质数
int n,cnt;//cnt存储结果个数
bool st[N];//表示当前数是否已经筛过(标记是否为质数) 

void get_prime(int x){
    for(int i=2;i<=x;i++){
        if(!st[i]) primes[cnt++]=i;//i是质数
        for(int j=2*i;j<=x;j+=i) st[j]=true;//不管i是不是质数，i的倍数一定不是质数
    }
}

int mian(){
    cin>>n;
    get_prime(n);
    cout<<cnt;
    return 0;
}
```



#### （4）筛质数--埃氏筛法

在上面朴素筛法的基础上，

其实不需要把全部数的倍数删掉，而**只需要删除质数的倍数即可**。

对于一个数p，判断其是否是质数，其实不需要把2到p-1全部数的倍数删一遍，只要删掉2到p-1之间的质数的倍数即可。因为，若p不是个质数，则其在2到p-1之间，一定有质因数，只需要删除其质因数的倍数，则p就能够被删掉。埃氏筛法筛选质数的时间复杂度大约为**O{nlog(logn)}**

##### AcWing 868. 筛质数

![img](assets\e84b98263697e2698582ef51bead4eee.png)

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int primes[N];//存储质数
int n,cnt;//cnt存储结果个数
bool st[N];//表示当前数是否已经筛过(标记是否为质数) 

void get_prime(int x){
    for(int i=2;i<=x;i++){
        if(!st[i]) {
            primes[cnt++]=i;//i是质数
            for(int j=2*i;j<=x;j+=i) st[j]=true;//只筛质数i的倍数
        }
        
    }
}

int mian(){
    cin>>n;
    get_prime(n);
    cout<<cnt;
    return 0;
}
```



#### （5）筛质数--线性筛法

大体思路和埃氏筛法一样，将合数用他的某个因数筛掉，其性能要优于埃氏筛法（在$10^{6}$下两个算法差不多，在$10^7$下线性筛法大概快一倍）核心思路是：**对于某一个合数n，其只会被自己的最小质因子给筛掉**

设置一个`primes`数组，存储质数（以下叙述用`pj`来表示`primes[j]`），**从2到n进行循环遍历**，用数组`st[]`标记是否为质数。**每次循环都对当前质数数组进行遍历，用其最小质因子筛除合数**

- 当`i % pj == 0`时：`pj` 一定是 `i` 的最小质因子，因为我们是从小到大枚举质数的，首先遇到的满足`i % p j == 0`的，**`pj `一定是` i `的最小质因子**，**并且`pj` 一定是`pj * i`的最小质因子**。比如，15 = 3 *5，15的最小质因子是3，则15的倍数中最小的数，其最小质因子同样是3的，15乘以最小质因子3，即45。

  **此时跳出循环**，因为此时`pj`是`i`的最小质因子，要利用大于`i`的数的最小质因数筛除那些大于`i`的数就必须保证`p[j]*i`中`p[j]`为最小质因子，但是由于继续进行的话`p[j]`大于`i`的最小质因子，因此`p[j]*i`的最小质因子不在是`p[j]`，所以要跳出循环

- 当`i % pj != 0`时：`pj `一定不是` i `的质因子,并且由于是从小到大枚举质数的，**那么 `pj` 一定小于 `i` 的全部质因子。那么` pj` 就一定是 `pj * i` 的最小质因子。**

因此，则**无论哪种情况，`pj`都一定是 `pj * i` 的最小质因子，`pj * i` 必为合数，标记筛除**

**线性筛法保证了，每个合数，都是被其最小质因子给删掉的，且只会被删一次**

##### AcWing 868. 筛质数

![img](assets\e84b98263697e2698582ef51bead4eee.png)

```c++
#include <iostream>
using namespace std;
const int N=1e6+10;
int primes[N];//存储质数
int n,cnt;//cnt存储结果个数
bool st[N];//表示当前数是否已经筛过(标记是否为质数) 

void get_prime(int x){
    for(int i=2;i<=n;i++){
        if(!st[i]) primes[cnt++]=i;//i是质数
        for(int j=0;primes[j]<=x/i;j++){
            st[primes[j]*i]=true;//筛除以primes[j]为最小质因数的合数
            if(i%primes[j]==0) break;//跳出这层循环
        }
    }
}

int mian(){
    cin>>n;
    get_prime(n);
    cout<<cnt;
    return 0;
}
```





### 2.约数

#### **（1）试除法：求一个数的所有约数**

利用试除法求一个数的所有约数，思路和判断和求质数的判定类似

**一个数N有一个约数d，那么N/d也必然是其约数**

约数都是成对出现的，只需要枚举1到$\sqrt{n}$即可，注意不要让一个约数加入两次

##### AcWing 869. 试除法求约数

![img](assets\dcff29ad55c9172f0203dcc779053eb9.png)

实现思路：这里使用vector<int>存储最终结果，便于进行最后的排序输出

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int n;

vector<int> get_divisors(int x){
    vector<int> res;
    for(int i=1;i<=x/i;i++){
        if(x%i==0){//找到一个约数 实际上就获得两个约数了
            res.push_back(i);
            if(i!=x/i) res.push_back(x/i);
        }
    }
    sort(res.begin(),res.end());
    return res;
}


int main() {
    cin>>n;
    while(n--)
    {
        int x;
        cin>>x;
        vector<int> res=get_divisors(x);
        for(auto t:res) cout<<t<<" ";
        puts("");
    }
 
    return 0;
}

```



#### **（2）求约数个数**

$$
将一个数N分解质因数为N=P_{1} ^{α_1} \times P_{2} ^{α_2} \times P_{3} ^{α_3} \dots \times P_{n} ^{α_n}
$$

$$
定理1：约数的个数为(α_{1} + 1) \times (α_{2} + 1) \times \dots \times (α_{n} + 1)
$$

对于定理1：N可以分解为以上n个质数的$α_i$次方的乘积，又因为N的每一个约数d都相当于在这n个pi中每个pi分别取$0到α_i$次，每一种选法的各个质因子相乘就是N的一个约数

eg：
$$
12=2^2*3^1
$$

- 取0 个 2 和 0 个 3 ， 得 约 数 1 
- 取1 个 2 和 0 个 3 ， 得 约 数 2 
- 取2 个 2 和 0 个 3 ， 得 约 数 4 
- 取0 个 2 和 1 个 3 ， 得 约 数 3 
- 取1 个 2 和 1 个 3 ， 得 约 数 6 
- 取2 个 2 和 1 个 3 ， 得 约 数 12 

**(2+1)*(1+1)=6种取法，则12有6个约数，即1，2，3，4，6，12**



#### **（3）求约数之和**

再在上面的基础上
$$
对于12=2^2*3^1来说，其约数之和为2^0*3^0+2^1*3^0+2^2*3^0+2^0*3^1+2^1*3^1+2^2*3^1
\\=(2^0+2^1+2^2)*(3^0+3^1)
$$
得到**定理2：约数之和**
$$
定理2：所有约数之和为(P_{1}^{0} + P_{1} ^ {1} + \dots + P_{1} ^{α_1}) \times (P_{2}^{0} + P_{2} ^ {1} + \dots + P_{2} ^{α_2}) \times \dots \times (P_{n}^{0} + P_{n} ^ {1} + \dots + P_{n} ^{α_n})
$$


##### AcWing 870. 约数个数

![img](assets\fc3d90baaf48efcaf5d458c17c676363.png)

**实现思路：**对每次输入的整数$a_i$都分解得到其质因数，并得到对应质因数的指数

- 使用一个**哈希表**来存储对应质因数与其指数的对应关系
  - `unordered_map<int,int> primes`：使用map实现哈希表，`i`对应质因数，`primes[i]`对应其指数

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long LL;
const LL mod=1e9+7;

int main(){
    int n;
    cin>>n;
    unordered_map<int,int> primes;//哈希表
    while(n--){
        int x;
        cin>>x;
        for(int i=2;i<n/i;i++){
            while(x%i==0){//得到质因数
                x/=i;
                primes[i]++;//记录质因数i 及其指数
            }
        }
        if(x>1) primes[x]++;
    }
    LL res=1;
    for(auto prime:primes){//遍历哈希表 根据定理 求约数个数
        res*=(prime.second+1)%mod;//题目要求取模
    }
    cout<<res;
}

```



##### AcWing 871. 约数之和

![img](assets\766904b7f6bb4409da38006c86dd9913.png)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long LL;
const LL mod=1e9+7;

int main(){
    int n;
    cin>>n;
    unordered_map<int,int> primes;//哈希表
    while(n--){
        int x;
        cin>>x;
        for(int i=2;i<n/i;i++){
            while(x%i==0){//得到质因数
                x/=i;
                primes[i]++;//记录质因数i 及其指数
            }
        }
        if(x>1) primes[x]++;
    }
    LL res=1;
    for(auto prime:primes){//遍历哈希表 根据定理 求约数之和
        int p=prime.first,a=prime.second;//得到各个质因数及其指数
        LL t=1;
        while(a--) t=(p*t+1)%mod;
        res*=t%mod;
    }
    cout<<res;
}

```





#### （4）最大公约数(GCD Greatest Common Divisor)

根据一个**显然的性质**：若d能整除b，d能整除a，即d是a和b的公约数，则d能整除`a*x+b*y`

求a和b的最大公约数-----**欧几里得算法（辗转相除法）**：`gcd(a,b) == gcd(b, a % b)`

**即a和b的最大公约数=b和a%b的最大公约数**，时间复杂度O(logn)

**证明：**可知`a mod b= a - [a/b]*b  `（[a/b]表示取整），这里简便表示，写为 `a mod b= a - c*b`

- 左到右：设a和b的最大公约数是k。由以上显然的性质可得k能整除a，k能整除b，则**k能整除（a-c*b）**
- 右到左：设b和a%b的最大公约数是k。因为k能整除b，k能整除a-c*b，则**k能整除`(a-c*b+c*b)`即a**

##### AcWing 872. 最大公约数

![img](assets\9abf4b0132006d917c053a26240a14be.png)

**实现思路：**当b不为0时，求b和a%b的最大公约数 ;当b为0时，即a和0的最大公约数，为a，因为0能整除任何数

```c++
#include <iostream>
using namespace std;

int gcd(int a,int b){
    return b? gcd(b,a%b):a;
}

int mian(){
    int n;
    cin>>n;
    while(n--){
        int x,y;
        cin>>x>>y;
        cout<<gcd(x,y)<<endl;
    }
    return 0;
}
```



### 3.欧拉函数

欧拉函数`phi(n)`：**得到1~n中与n互质的数的个数**

例如：phi(6) = 2，1 - 6 中与 6 互质的数为 1、5

a，b互质就是`gcd(a,b) = 1`

**如何求解欧拉函数？**

对于一个数N，可以分解质因数为
$$
N = P_{1}^{k_{1}} \times P_{2}^{k_{2}} \times \dots \times P_{n}^{k_{n}}，则phi(N) = N \times (1 - \frac{1}{P_{1}}) \times (1 - \frac{1}{P_{2}}) \times \dots \times (1 - \frac{1}{P_{n}})
$$
比如 6=2×3，则phi(6) = 6 * (1 - 1/2) * (1-1/3) = 2，即1~6中有两个与6互质的数

**证明**：利用**容斥原理**
$$
①从1 - N 中 去除其全部质因子P_{1} \dots P_{n}的所有倍数，那么还剩下S_{1} = N - \frac{N}{P_{1}} - \frac{N}{P_{2}} - \dots - \frac{N}{P_{n}}个数\\
②有的数既是P_{i}的倍数又是P_{j}的倍数，因此被减了两遍，需要加一次回来，S_{2} = S_{1} + \frac{N}{P_{1}\times P_{2}}+ \frac{N}{P_{1}\times P_{3}} + \dots + \frac{N}{P_{1}\times P_{n}} + \frac{N}{P_{2}\times P_{3}} + \dots +\frac{N}{P_{2}\times P_{n}} + \dots + \frac{N}{P_{n-1}\times P_{n}}\\
③有的数是P_{i}、P_{j}、P_{k}的倍数，在第一步减了三次，在第二步加了三次，相当于没加也没减，因此需要减掉一次，因此S_{3} = S_{2} - \frac{N}{P_{i}\times P_{j} \times P_{k}}，[i,j,k]是1 - n的一组排列
$$
以此类推到第n步，化简就是上边的公式`phi(N)`的求解公式



**欧拉函数的应用：**

**欧拉定理**：
$$
若a与n互质，那么有a^{\phi(n)}  mod \ n=1
$$
**费马定理：**
$$
若a与p互质，p是质数，那么有a^{\phi(p)}  mod\ p  = a^{p-1}  mod\ p = 1
$$


#### 定义求欧拉函数

##### **AcWing 873. 欧拉函数**

![img](assets\a84f9b05f88cc8947f92b8faae139169.png)

实现思路：得到质因数，然后按公式相乘

注意：每次得到质因数i，应按照公式乘以（1-1/i)，但为避免小数，先除再乘，即`(1-1/i)=/i*(i-1)`

```c++
#include<iostream>
using namespace std;
int mian(){
    int n;
    cin>>n;
    while(n--){
        int a;
        cin>>a;
        int res=a;
        for(int i=2;i<n/i;i++){
            if(a%i==0){//得到质因数
                res=res/i*(i-1);
                while(a%i==0) a/=i;//将a除净
            }
        }
        if(a>1) res=res/a*(a-1);//如果a还大于1 a为一个质因数
        cout>>res>>endl;
        return 0;
    }
}
```

#### **筛法求欧拉函数**

**利用质数的线性筛法求1-n的欧拉函数**

由于在线性筛法的执行过程中，对于质数会保留，对于合数会用其最小质因子筛掉。所以线性筛法是会访问到所有数的。而根据上面的推导，在遇到每种情况时，我们都能求出欧拉函数

- 当i是质数：`phi(i)=i-1`，因为i为质数，那么前i-1个都与其互质

- 当i是合数：

  某个合数一定是被`pj * i` 给筛掉的，我们就在筛他的时候求他的欧拉函数值

  1. 如果 `i % pj == 0` ,那么`pj`就是`i`的某个质因数，那么`pj*i`和`i`的**质因数组合完全相同**，根据欧拉函数公式，所以`phi(pj *i) = pj * phi(i)`
  2. 如果`i % pj != 0`，pj不是i的质因数，那么`pj*i`的质因数组合就是**在`i`的质因数组合基础上加了一个质数`pj`**，根据欧拉函数公式，所以`phi(pj * i) = pj * phi(i) *(1 - 1/pj) = (pj - 1) * phi(i)`

##### AcWing 874. 筛法求欧拉函数

![img](assets\451bbc891ca29a4adc93587bb4d46e0e.png)

在质数线筛法的基础上修改

```c++
#include <iostream>
using namespace std;
typedef long long LL;
const int N=1e6+10;
int primes[N],phi[N],cnt;//phi[N]存储欧拉函数结果
bool st[N];

LL get_euler(int n){
    phi[1]=1;//1只有其本身与其互质
    LL res=0;
    for(int i=2;i<=n;i++){
        if(!st[i]) {//i为质因数
            primes[cnt++]=i;
            phi[i]==i-1;//前面i-1都与其互质
        }
        for(int j=0;primes[j]<=n/i;j++){
            st[primes[j]*i]=true;
            if(i%primes[j]==0){
                phi[primes[j]*i]=primes[j]*phi[i];
                break;
            }
            phi[primes[j]*i]=phi[i]*(primes[j]-1);
        }
    }
    for(int i=1;i<=n;i++)
        res+=phi[i];
    return res;
}


int main(){
    int n;
    cin>>n;
    cout<<get_euler;
}

```



### 4.快速幂

**作用**：可以快速的求出**$ a^{k} mod \ 𝑝$** 的值，时间复杂度是O(logk)

核心思路：**反复平方法**
$$
①预处理出：a^{2^{0}} mod\ p、a^{2^{1}} mod\ p、a^{2^{2}} mod\ p、\dots、a^{2^{\log_{2}{k} }} mod\ p 一共log_{2}{k}个数\\
对于这预处理出的数，观察可以发现：a^{2^{1}}=(a^{2^{0}})^2、a^{2^{2}}=(a^{2^{1}})^2、a^{2^{3}}=(a^{2^{2}})^2、\dots、a^{2^{log_{2}{k}}}=(a^{2^{log_{2}{k}-1}})^2.\\即每个数是前面一个数的平方\\
\\
②对于a^{k}，可将 a^{k} 拆成 a^{k} = a^{2^{x_{1}}} \times a^{2^{x_{2}}} \times \dots \times a^{2^{x_{t}}} = a^{2^{x_{1}}+2^{x_{2}}+\dots+2^{x_{t}}}\\
可得到k=2^{x_{1}}+2^{x_{2}}+\dots+2^{x_{t}}\ ，即k为log_{2}{k}个数之和\\

那么a^{k} mod\ p=(a^{2^{x_{1}}}mod \ d)*(a^{2^{x_{2}}}mod \ d)...(a^{2^{x_{t}}}mod \ d)\\
再由①中得到的每个数是前一个数的平方，故a^{k} mod\ p
=(a^{2^{x_{1}}}mod \ d)*[(a^{2^{x_{1}}}mod \ d)]^2 mod\ d\ * ...*\\(前一个结果的平方mod\ d)\\
\\③对于怎么得到a^{k}=a^{2^{x_{1}}+2^{x_{2}}+\dots+2^{x_{t}}}，简单，直接取k的二进制数。
\\ \\
如求4^5 mod \ 10，则k=(101)_2，那么4^{5}=4^{(101)_{2}}=4^{2^{0}}\times 4^{2^{2}}
\\而4^{2^{0}} mod\ 10=4，4^{2^{1}}mod\ 10=4^2 mod \ 10=6，4^{2^{2}}mod\ 10=6^2 mod \ 10=6，\\(这里就体现了后面的结果为前一个结果的平方再取模)
\\得到最终结果即为4^5 mod \ 10=4*6 \ mod \ 10=4
$$

#### **AcWing 875. 快速幂**

![img](assets\badeaf4e554c24642cf096d7878412fb.png)

```c++
#include <iostream>
using namespace std;
typedef long long LL;//a可能超出int

//a^k % p
LL qmi(LL a,int k,int p){
    int res=1;//记结果
    while(k){//循环得到k的每个二进制位 
        if(k & 1) res=(LL)res * a % p;//当前二进制位为1时
        k=k>>1;//k右移一位 找下一个二进制位
        a=(LL)a*a % p;//平方一下 再mod p
    }
    return res;
}

int main(){
    int n;
    cin>>a;
    while(n--){
        int a,k,p;
        cin>>a>>k>>p;
        cout<<qmi(a,k,p)<<endl;
    }
    
    return 0;
}
```



#### AcWing 876. 快速幂**求逆元**

![img](assets\a8ea462d396dca04605ad2a7a77506ca.png)

**实现思路：**

- 由(a/b)mod m恒等a*x mod m===>`(b*x) mod m = 1 `，x为b的逆元
- **结合费马定理**(欧拉函数的应用)：b与m互质，且m为质数，则b^(m-1) mod m=1
- 上面两式结合：得b的逆元b^(m-2)，则**模m乘法逆元`x=b^(m-2)%m`**，**本质上就是求快速幂，但多了一个要求：b和m互质**
- 注意：当b和m不互质时，无解；否则必存在逆元

```c++
#include <iostream>
using namespace std;
typedef long long LL;

//求快速幂
LL qmi(LL a,int k,int p){
    LL res=1;
    while(k){//对每个二进制位处理
        if(k & 1) res=(LL)res*a%p;
        k=k>>1;
        a=(LL)a*a%p;
    }
    return res;
}

int mian(){
    int n;
    cin>>n;
    while(n--){
        int a,p;
        cin>>a>>p;
        int res=qmi(a,p-2,p);//根据推理得到的公式
        if(a%p) cout<<res<<endl;//如果a和p互质
        else puts("impossible");
    }
    
    
}
```



### **5.扩展欧几里得算法**

回忆：求**最大公约数**中学过**欧几里得算法（辗转相除法）**：`gcd(a,b) == gcd(b, a % b)`

**裴蜀定理**：对于任意正整数a，b，那么一定存在非零整数x，y，使得`ax+by=gcd(a,b)`

> 证明：令 gcd(a,b) = c ，则a一定是c的倍数，b也一定是c的倍数，那么ax+by也一定是c的倍数，那么可以凑出最小的倍数就是1倍，即ax+by=gcd(a,b)

**扩展欧几里得算法**：就是求解上面裴蜀定理中**a和b的系数x,y**

**具体怎么求解x，y？**`ax+by=gcd(a,b)`

- 首先若有一个数为0，假设b=0，那么显然gcd(a,0)=a，则可得x=1，y=0
- 若a，b不为0，结合欧几里得算法`gcd(a,b) == gcd(b, a % b)`，那么`ax+by=gcd(a,b)`===>`b*x+(a%b)*y=gcd(a,b)`，但在递归的过程中a的系数由x变为y，b的系数由y变为x，所以**在递归过程的传入参数时因调换x和y的位置**，变为`b*y+(a%b)*x=gcd(a,b)`，下面的代码会体现
- 设gcd(a,b)=d，那么继续推理得到

​	<img src="assets\image-20240729114041900.png" alt="image-20240729114041900" style="zoom:50%;" />

即可求出系数

- 注意x和y可能是不唯一的

#### AcWing 877. 扩展欧几里得算法

![img](assets\235171f5a9ee28dc7adf91ee3ac82706.png)

实现思路：就是在欧几里得算法的基础上修改，增加两个传入参数x，y

```c++
#include <iostream>
using namespace std;

//这里返回的是最大公约数
int exgcd(int a,int b,int &x,int &y){//这里x和y使用引用，后续直接输出答案
    if(!b){//b=0
        x=1,y=0;
        return a;
    }
    int d=exgcd(b,a%b,y,x);//注意y和x位置
    y-=a/b*x;
    return d;
    
}


int main() {
    int n;
    cin>>n;
    while(n--)
    {
        int a,b,x,y;
        cin>>a>>b;
        exgcd(a,b,x,y);
        cout<<x<<" "<<y<<endl;
    }
    return 0;
}

```

#### AcWing 878. 线性同余方程

**扩展欧几里得的应用**：求解线性同余方程

![img](assets\5c5fe9502b215cbc733dd586a8399956.png)

**实现思路：**对于`(a*x) mod m=b`，求解x。

- 恒等变形可以得`a*x=m*y'+b`===>`a*x-m*y'=b`===>**`a*x+m*y=b`**，此时就形似扩展欧几里得算法，则由裴蜀定理可知`a*x+m*y=gcd(a,m)`必然有解，假如**b是gcd(a,m)的倍数**，即必然存在这样的x和y，使所求等式有解，只需等式两边扩大相应的倍数即可。此时**`x=x*(b/gcd(a,m))`**
- 这里最后**`x*(b/gcd(a,m))%m`**是为了得到最小的解，因为x的通解有无数个，防止超出int范围，就取一个最小的
- 假如题目要求x的**最小正整数解**，则**[x%(b/gcd)+b/gcd]%(b/gcd)**
- 其他理解：[AcWing 878. 线性同余方程关于最后结果处理的证明 - AcWing](https://www.acwing.com/solution/content/11231/)

```c++
#include <iostream>
using namespace std;
typedef long long LL;

//扩展欧几里得 返回最大公约数
int exgcd(int a,int b,int &x,int &y){
    if(!b){//b=0
        x=1,y=0;
        return a;
    }
    int d=exgcd(b,a%b,y,x);//记得调换x和y的顺序
    y-=a/b*x;
    return d;
    
}

int main(){
    int n;
    cin>>n;
    while(n--){
        int a,b,m,x,y;
        cin>>a>>b>>m;
        int d=exgcd(a,m,x,y);
        if(b%d==0) cout<<(LL)x*(b/d)%m<<endl;//取得最小的解
        else puts("impossible");
    }
    return 0;
}
```



### 6.中国剩余定理

$$
有k个两两互质的数m_{1}、m_{2}、\dots 、m_{k}\\
给定线性同余方程组\ x\equiv a_{1}(mod\ m_{1})、x\equiv a_{2}(mod\ m_{2})、… 、x\equiv a_{k}(mod\ m_{k})\\
\\求x解法：\\
令M=m_{1}\times m_{2}\times \dots \times m_{k}
，M_{i}=\frac{M}{m_{i}}，用M_{i}^{-1}表示M_{i} 模的逆 
\\(求M_{i}^{-1}：M_{i} \times M_{i}^{-1} \equiv 1 (mod \ m_{i})，实质就是用扩展欧几里得求一个特殊的线性同余方程:a \times x\equiv 1 (mod \ m))
\\
则x=a_{1}\times M_{1} \times M_{1}^{-1} + a_{2}\times M_{2} \times M_{2}^{-1} + \dots + a_{k}\times M_{k} \times M_{k}^{-1}
\\可以验证x为该线性同余方程组的解（略）
$$

#### **AcWing 204. 表达整数的奇怪方式**

![img](assets\cd8b2d527d2b34dcb2bdf51089e0321d.png)

**实现思路**：

**求解推导**
$$
1.对于两个式子，考虑将其合并：
\\  x\equiv \ m_1 (\% a_1 )
\\  x\equiv \ m_2 (\% a_2 )
\\则有：x=k_1*a_1+m_1,x=k_2*a_2+m_2,
\\进一步联立上面两式：k_1*a_1+m_1=k_2*a_2+m_2==>\mathbf {k_1*a_1+k_2*(-a_2)=m_2-m_1...①}
\\也就是我们要找到一个最小的k_1,k_2，使得等式成立(因为要求x最小，而a和m都是正数)
\\ \\2.对a_1和a_2使用扩展欧几里得得到最小公约数d=gcd(a_1,-a_2)
\\即\mathbf {k_1*a_1+k_2*a_2=d=gcd(a_1,-a_2)...②}
\\此时对于①和②式就出现了之前求线性同余方程的情况，即m_2-m_1是d的倍数时(d|m_2-m_1)方程有解
\\②式两边乘\frac{m_2-m_1}{d}就转化为①式
\\即\mathbf {k_1*a_1* \frac{m_2-m_1}{d} +k_2*a_2* \frac{m_2-m_1}{d}=m_2-m_1}
\\ \\3.有解的情况下得到一组解:
\\ \mathbf {k_1=k_1* \frac{m_2-m_1}{d} ,\ k_2=k_2* \frac{m_2-m_1}{d}},注意这里k就表示特解
\\ \\4.找最小正整数解：模尽倍数后，剩下的余数就是最小正整数解
\\k_1=k_1\ \% \ abs(\frac{a_2}{d}),k_2=k_2\ \% \ abs(\frac{a_1}{d})
\\ \\5.找满足方程的所有解
\\将一组方程的解带回方程：k_1*a_1+k_2*a_2=c
\\等式左边加减\frac{a_1*a_2}{d},方程右边不变，可得：
\\k_1*a_1+k_2*a_2+a_1*\frac{a_2}{d}-a_2*\frac{a_1}{d}=(k_1+\frac{a_2}{d})*a_1+(k_2-\frac{a_1}{d})*a_2=c
\\ \mathbf {于是得到所有解集：k_1=k_1+\frac{a_2}{d}*k,k_2=k_2-\frac{a_1}{d}*k}

\\ \\6.再代入k_1,k_2得新的x为：x=(k_1+\frac{a_2}{d}*k)*a_1+m_1
\\=k_1*a_1+m_1+\frac{a_1*a_2}{d}*k
\\令新的a_1^{'}=\frac{a_1*a_2}{d},新的m_1^{'}=k_1*a_1+m_1
\\那么x=a_1^{'}*k+m_1^{'},这时就又回到了第一步，至此完成两个式子的合并,
\\后续再来一个a_3,m_3同理继续合并，n个式子进行n-1次合并得到最终式
$$
其他理解：[AcWing 204. 表达整数的奇怪方式 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/2628/)

```c++
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
typedef long long LL;
int n;

//扩展欧几里得求系数 返回最大公约数d
LL exgcd(LL a,LL b,LL &x,LL &y){
    if(b==0){
        x=1,y=0;
        return a;
    }
    LL d=exgcd(b,a%b,y,x);//记得调换x和y的位置
    y-=a/b*x;
    return d;
}

//求a%b=a的最小正整数解
LL inline mod(LL a,LL b){
    return (a%b+b)%b;
}

int main(){
    cin>>n;
    LL a1,m1;
    cin>>a1>>m1;
    for(int i=1;i<n;i++){//n-1次合并
        LL a2,m2,k1,k2;
        cin>>a2>>m2;
        LL d=exgcd(a1,-a2,k1,k2);
        //如果m2-m1不能整除d 则无解结束
        if((m2-m1)%d) {puts("-1");return 0;}
        //否则有解
        k1=mod(k1*(m2-m1)/d,abs(a2/d)); //模尽得到最小正整数解
        //更新m1,a1 继续下轮合并
        m1=k1*a1+m1;
        a1=abs(a1/d * a2)
    }
}

```



## （二）高斯消元

高斯消元能在O(𝑛^3)的时间复杂度内**求解**n个方程，n个未知数的**多元线性方程组**，即
$$
a_{11}x_{1}+a_{12}x_{2}+a_{13}x_{3}+\dots +a_{1n}x_{n} = b_{1}\\a_{21}x_{1}+a_{22}x_{2}+a_{23}x_{3}+\dots +a_{2n}x_{n} = b_{2}\\ \dots \\ a_{n1}x_{1}+a_{n2}x_{2}+a_{n3}x_{3}+\dots +a_{nn}x_{n} = b_{n}
$$
对增广矩阵做初等行列变换，变成一个**行最简阶梯型矩阵**（线性代数）

- 对某一行（列）乘以一个非零的数
- 交换两行（列）
- 将一行（列）的若干倍加到另一行（列）

解的情况有三种

- 无解，系数矩阵秩不等于增广矩阵的秩
- 有无穷多解，系数矩阵秩等于增广矩阵的秩，且小于n
- 有唯一解，系数矩阵秩等于增广矩阵的秩，且等于n

**高斯消元算法步骤**：

从列开始，循环枚举处理每一列

- 找到该列绝对值（使用<cmath>中的fabs函数）最大的一行，如果最大的绝对值为0，说明当前列已经化简好了，此重循环（列）跳出进行下一列处理

  > 这里就可以区分出后续的r是否++，以此判断唯一解 与 无穷解+无解

- 将这一行换到最上面

  > 循环交换两行的各个元素

- 将该行第一个数变成1

  > 每一行的各个元素，从后往前除以该行的首元素

- 用当前行将下面所有行的当前列消成0

  > 用下面每一行每一列的元素-当前行的每一个元素*下面每一行当前列的元素

- 固定该行，处理下一行，处理的行数r++

  > 即当前行的位置就固定了

#### AcWing883. 高斯消元解线性方程组

![img](assets\172a1ff527b423763da88af95cbcf85b.png)

注：

- 本题采用fabs即浮点绝对值来取绝对值，在判断是否为0时，应为小于一个足够小的数即为0
- 用每一行的最后一列存储解的值
- 最后有唯一解的求解原因更详细说明[AcWing 883. 高斯消元解线性方程组 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/155868/)

```c++
#include <iostream>
#include <algorithm>
#include <cmath> //fabs的头文件
using namespace std;
const int N=110;
const double eps=1e-6;//绝对值小于10^-6即为0

int n;
double a[N][N];//存储增广矩阵

//高斯消元法解线性方程组 
int gauss(){
    int c,r;//c代表当前处理列，r代表当前处理行
    for(c=0,r=0;c<n;c++){//按列处理
        int t=r;//先找到当前这一列，绝对值最大的一个数字所在的行号
        for(int i=r;i<n;i++)
            if(fabs(a[i][c])>fabs(a[t][c]))
                t=i;
        //判断当前列最大绝对值为 0 ，那么这一列所有数都是 0，当前列无需继续化简，结束这轮循环，进入处理下一列
        if(fabs(a[t][c])<eps) continue; 
        
        //将当前行换到上面第r行（注意随着每一次完整处理都会固定一行，则当前行是换到固定行的下面一行，第一次就换到第一行）
        for(int i=c;i<=n;i++) swap(a[t][i],a[r][i]);//从当前行的第c列开始换
        
        //将该行的第c列的数改为1 该行的元素从后往前改
        for(int i=n;i>=c;i--) a[r][i]/=a[r][c];
        
        //将下面的行的该列元素消为0
        for(int i=r+1;i<n;i++)
            if(a[i][c]>eps)//非0才操作 若已经是0的行就没必要操作
                //从后往前，当前行的每个数字，都减去对应列 * 行首非0的数字，这样就能保证该行c列数字是0;
                for(int j=n;j>=c;j--)
                    a[i][j]-=a[r][j]*a[i][c];
        
        r++;//表示已经固定了一行 寻找固定下一行 中间可能会跳行
    }
    
    if(r<n){
        for(int i=r;i<n;i++)
            if(fabs(a[i][n])>eps)//若最后一列的元素非0 表示系数矩阵的秩与增广矩阵的秩不等 无解
                return 2;
        return 1;//否则系数矩阵的秩与增广矩阵的秩相等 且<n 无穷多解
    }
    
    //r=n 有唯一解 求解
    for(int i=n-1;i>=0;i--)//从最后一行开始倒推解xi
        for(int j=i+1;j<n;j++)
            /*
            j=i+1,i的下一行 下一列的数
            a[i][j]表示xj的系数，a[j][n]表示xj
            */
            a[i][n]-=a[i][j]*a[j][n];//解存储在最后一列
    return 0;
}


int main(){
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<=n;j++)
            cin>>a[i][j];
    int t=gauss();
    if(t==0){//有唯一解
        for(int i=0;i<n;i++) printf("%.2lf\n", a[i][n]);
    }else if(t==1) puts("Infinite group solutions");//无穷解
    else puts("No solution");//无解
}
```



#### AcWing884. 高斯消元解异或线性方程组

![img](assets\6428a7939e73be38238d8ef045093a08.png)

**实现思路**：基本思路和上题求解一般的线性方程组一致，只是区别在与本题未知量之间是异或运算，更适合电脑的运算。

**高斯消元得到上三角矩阵**
    1、枚举列
    2、找到当前列的非零行
    3、非零行换到剩余行的最上面
    4、剩余行中除去最上面一行，下面所有行的当前列都消零 
**判断解的种类**
    1、无解
    2、无穷多解
    3、唯一解

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=110;
int a[N][N];
int n;

int gauss(){
    //消成阶梯型矩阵
    int r,c;
    for(r=c=0;c<n;c++){//1.枚举列开始
        //2.找到剩余行中的非零行 从r行开始(因为上轮循环r++或跳过)
        int t=r;
        for(int i=r;i<n;i++)
             if(a[i][c]){
                t=i;
                break;
            }
        //跳出循环
        if(a[t][c]==0) continue;//表示当前列没有非零行 直接进入下一列循环
        
        //3.找打了满足的非零行,与剩余行的最上面交换
        for(int i=c;i<=n;i++) swap(a[i][r],a[t][i]);
        
        //4.剩余行中的第c列都消为0
        for(int i=r+1;i<n;i++)
            if(a[i][c]){//非0就消为0
                for(int j=c;j<=n;j++)
                    a[i][j]^=a[r][j];//做异或
            }
        r++;//已经固定一行 下一轮
    }
    
    //5.判断解的情况
    if(r<n){
        for(int i=r;i<n;i++)
            if(a[i][n]) return 2;//无解
        return 1;//无穷解
    }else{//唯一解
        for(int i=n-1;i>=0;i--)
            for(int j=i+1;j<n;j++)
                a[i][n]^=a[i][j]*a[j][n];//倒推求解
        return 0;
    }
    
}

int main(){
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<=n;j++)
            cin>>a[i][j];
    int res=gauss();
    if(res==0){//唯一解
        for(int i=0;i<n;i++) cout<<a[i][n]<<endl;
    }else if(res==1) puts("Multiple sets of solutions");//无穷多解
    else puts("No solution");//无解
    return 0;
}
```



## （三）组合计数

$$
从a个元素中选择b个，有多少种取法C_{a}^{b} = \frac{a\times(a-1)\times\dots\times(a-b+1)}{1\times2\times3\times\dots\times b} 
\\=\frac{a!}{b!\times(a-b)!}
\\= C_{a-1}^{b} + C_{a-1}^{b-1}
$$

### 组合计数 I

#### AcWing 885. 求组合数 I

![img](assets\f02942edc68b0670a8e60e435fd07aef.png)

**实现思路**：询问10000次，数字范围2000，直接使用**递推式**

- 递推式：
  $$
  C_{a}^{b} = C_{a-1}^{b} + C_{a-1}^{b-1}
  $$
  其实就是动态规划DP，O(N^2)

> 证明：
> $$
> 从a个元素中存在某个元素，在选择b个元素过程中，共两种情况，包含这个元素和不包含这个元素
> \\如果要选的b个元素包含这个元素，那么就是从剩下的a-1个元素中选b-1个，即C_{a-1}^{b-1}，
> \\如果不包含这个元素，就是从a-1个元素中选择b个，即C_{a-1}^{b}，综合两种情况就是C_{a}^{b} = C_{a-1}^{b} + C_{a-1}^{b-1}
> $$

- 因为**数字范围就2000**，直接先**预处理出2000范围内的所有组合数**，使用二维数组存储`c[][]`,后续只需调用就行
- 注意结果要取模，防止结果超出int范围

```c++
#include <iostream>
using namespace std;
const int N=2010,mod=1e9+7;
int c[N][N];

//预处理出这2000范围内的所有组合数
void init(){
    for(int i=0;i<N;i++)
        for(int j=0;j<=i;j++)
            if(!j) c[i][j]=0;//若j为0
    		else c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;//注意取模，防止结果超出int范围
}

int mian(){
    int n;
    cin>>n;
    init();
    while(n--){
        int a,b;
        cin>>a>>b;
        cout<<c[a][b]<<endl;;
    }
    return 0;
}

```



### 组合计数 II

#### AcWing 886. 求组合数 II

![img](assets\bf6ef8706f6afd231e68ef8f01eab98e.png)

**实现思路：**相比组合数I，这里数据范围增大到$10^5$，在预处理的过程中不能直接像组合数I一样求，否则时间复杂度过高$10^{10}$，这里使用**公式+乘法逆元**的方式预处理得到阶乘
$$
公式C_{a}^{b} =\frac{a!}{b!\times(a-b)!}
$$

- 考虑到计算过程存在取模且公式存在分式，则需要将除法转化为乘法，因为除法取模会得到错误答案，所以**转化为乘法再做取模运算**，这时就想到**乘法逆元（因为本题模数1e9+7是质数，所以可以使用快速幂求逆元，费马定理b^(m-2)。注意若没有对质数取模，则只能用扩展欧几里得算）**，对分母两个数分别取逆元再乘以分子，即可得到答案

  > **明明是除法为什么要特意转换成乘法？**
  >
  > 那是因为这道题目最后是要求余的，模运算与基本四则运算有些相似，但是除法例外。其规则如下：
  > (a + b) % p = (a % p + b % p) % p
  > (a - b) % p = (a % p - b % p) % p
  > (a * b) % p = (a % p * b % p) % p
  > 但对于除法却不成立，即(a / b) % p 不等于 (a % p / b % p) % p 。
  >
  > [AcWing 886. 关于求组合数 II的一些疑问 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/6539365/)

- 设置一个数组fact[]，表示分子阶乘；数组infact[]，表示分母阶乘的逆元，最后通过`fact[a]*infact[b]*infact[a-b]`得到组合数结果

```c++
#include <iostream>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N=1e4+10,mod=1e9+7;

LL fact[N],infact[N];//分子阶乘，和分母阶乘的逆元

//求快速幂
LL qmi(LL a,LL k,LL p){
    LL res=1;
    while(k){
        if(k & 1) res=(LL)res*a%p;
        k=>>1;
        a=(LL)a*a%p;
    }
    return res;
}

int main(){
    fact[0]=infact[0]=1;
    for(int i=1;i<N;i++){
        fact[i]=fact[i-1]*i%mod;//阶乘结果
        infact[i]=qmi(fact[i],mod-2,mod);//阶乘的逆元
    }
    int n;
    cin>>n;
    while(n--)
    {
        int a,b;
        cin>>a>>b;
        cout<<(LL)fact[a]*infact[b]%mod*infact[a-b]%mod<<endl;
    }
    return 0;
}
```

### 组合计数 III

#### AcWing 887. 求组合数 III

![img](assets\2255cf1d8ba4ace150aa7a565b432080.png)

**实现思路：**询问很少，但**数据范围巨大**1e18，使用**卢卡斯定理**，时间复杂度为`O(p*logN*logp)`，1<p<1e5

- **卢卡斯（lucas）定理：**
  $$
  C_{a}^{b} \equiv C_{a\ mod \ p}^{b\ mod \ p} \times C_{a/p}^{b/p} \pmod{p}
  $$

  > $$
  > 简要证明：\\
  > 把a和b转换为p进制表示
  >    
  >     \\a = a_{k}p^{k}+a_{k-1}p^{k-1}+\dots+a_{0}p^{0}
  >     
  >     \\b= b_{k}p^{k}+b_{k-1}p^{k-1}+\dots+b_{0}p^{0}
  >    
  >     \\生成函数(1+x)^{p} = C_{p}^{0}*1 + C_{p}^{1}*x^1 + C_{p}^{2}*x^2 + \dots+C_{p}^{p}*x^p \equiv 1 + x^p \pmod{p} 
  >     
  >   \\所以有(1+x)^a = ((1+x)^{p^{0}})^{a_{0}} \times ((1+x)^{p^{1}})^{a_{1}} \times ((1+x)^{p^{2}})^{a_{2}}\times \ dots \times ((1+x)^{p^{k}})^{a_{k}} = (1+x)^{a_{0}} \times (1+x^{p^{1}})^{a_{1}} \times (1+x^{p^{2}})^{a_{2}}\times \dots \times (1+x^{p^{k}})^{a_{k}}
  >     
  >     \\\\对比等式(1+x)^a = (1+x)^{a_{0}} \times (1+x^{p^{1}})^{a_{1}} \times (1+x^{p^{2}})^{a_{2}}\times \dots \times (1+x^{p^{k}})^{a_{k}} \ 左右两边x^{b}项的系数，C_{a}^{b} \equiv C_{a_{k}}^{b_{K}} \times C_{a_{k-1}}^{b_{k-1}} \times\dots \times C_{a_{0}}^{b_{0}}\pmod{p}
  > $$

  **lucas函数：**函数的参数为a,b，若**a<p且b<p**则说明a,b足够小可以通过组合数函数C(a,b)直接计算，否则返回**C（a%p,b%p）*lucas(a/p,b/p)**（因为a,b除p可能依然很大，当足够小时就像上述情况直接返回组合数函数）

- 注意这里求组合数C，与上题不同，无需枚举处理所有组合数，直接用基本公式：
  $$
  C_{a}^{b}=\frac{a*(a-1)...*(a-b+1)}{b!}
  $$

​	采用类似双指针算法，指针i由后向前遍历到a-b+1，j由1向前遍历到b

​	**然后再利用乘法逆元（快速幂求逆元），将除法转化为乘法，取模**

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long LL;
int p;//全局变量 后面函数的参数无需再传入p

//求快速幂
LL qmi(int a,int k){
    LL res=1;
    while(k){
        if(k & 1) res=(LL)res*a%p;
        k=>>1;
        a=(LL)a*a%p;
    }
    return res;
}

//基本公式求组合数（利用乘法逆元）
LL C (int a,int b){
    if(a<b) return 0;
    LL x=1,y=1;//x是分子，y是分母
    for(int i=a,j=1;j<=b;i--,j++){
        x=(LL)x*i%p;
        y=(LL)y*j%p;
    }
    return x*(LL)qmi(y,p-2,p)%p;//快速幂乘法逆元
}

//卢卡斯定理
LL lucas(int a,int b){
    if(a<p && b<p) return C(a,b);//a和b都小于p，直接用C求就可以
    return (LL)C(a%p,b%p)*C(a/p,b/p)%p;
}


int main()
{
    int n;
    cin>>n;
    while(n--)
    {
        LL a,b;
        cin>>a>>b>>p;
        cout<<lucas(a,b)<<endl;
    }
}
```



### 组合计数 IV

当我们需要求出组合数的真实值，而非对某个数的余数时，**分解质因数**的方式比较好用，同时需要用到高精度乘法

#### AcWing 888. 求组合数 IV

![img](assets\b191e75c62c6ecbe71b5dc8af0e27e5f.png)

**实现思路：**

- 使用组合数公式
  $$
  C_{a}^{b} =\frac{a!}{b!\times(a-b)!}
  $$
  
- 用**线筛法求出对于公式中最大的数a之前的所有的质数**，之后在得到在`a!`的中所包含的每个质因子p对应的幂次
  $$
  a!中质因子p的幂次=\frac{a}{p}+\frac{a}{p^2}+.....\frac{a}{p^n}，向下取整
  $$

- 同理得到`b!`和`(a-b)!`中对应p的质因子的幂次，然后**a!中p的幂次-b!中p幂次-(a-b)!中p的幂次=结果中p的幂次**，以此类推得到结果中每个质因子pi对应的幂次
  $$
  最终结果C_{a}^{b}=p_{1}^{α1}*p_{2}^{α2}*p_{3}^{α3}*....*p_{k}^{αk}
  $$
  
- 再对结果进行一个高精度的乘法得到最终结果

```c++
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
const int N=5010;
int primes[N],sum[N];//存储质数,结果中对应质因子的幂次
bool st[N];
int cnt;

//得到1~x中的质数   线筛法
void get_prime(int x){
    for(int i=2;i<=x;i++){
        if(!st[i]) primes[cnt++]=i;
        for(int j=0;primes[j]<=x/i;j++){
            st[primes[j]*i]=true;
            if(i%primes[j]==0) break;
        }
    }
}

//得到n的阶乘中质因子p的幂次
int get(int n,int p){
    int res=0;
    while(n){
        res+=n/p;
        n/=p;
    }
    return res;
}

//高精度乘法（第一章）
vector<int> mul(vector<int> &res,int b){
    int t=0;//表示进位
    vector<int> c;
    for(int i=0;i<res.size() || t;i++){
        if(i<res.size()) t+=res[i]*b;
        c.push_back(t%10);
        t/=10;
    }
    return c;
}


int mian(){
    int a,b;
    cin>>a>>b;
    get_primes(a);//得到所有质数
    for(int i=0;i<cnt;i++){
        int p=primes[i];//得到质因子
        sum[i]=get(a,p)-get(b,p)-get(a-b,p);//得到结果对应质因子的幂次
    }
    vector<int> res;
    res.push_back(1);
    
    //循环高精度乘法得到结果
    for(int i=0;i<cnt;i++)
        for(int j=0;j<sum[i];j++)//对应幂次循环
            res=mul(res,primes[i]);
    
    //从数组末尾开始输出结果高位
    for(int i=res.size()-1;i>=0;i--) cou<<res[i];
}
```



### 卡特兰数

#### AcWing 889. 满足条件的01序列

![img](assets\468c76afcfea3218705e6f8aa077f81d.png)

**实现思路**：转化为求图中**从原点到目标点的路径中符合某种条件的路径的方案数**

<img src="assets\be98abb9b60fad7e5fbcf8f9d34219a7.png" alt="img" style="zoom:67%;" />

假设有6个0，6个1，设0表示向右走一步，1表示向上走一步，则路径是从原点（0,0）到点(6,6)，需要12步。根据题目要求**前缀中0的个数大于1的个数**，转化为所走路径要满足的**条件**就是：**路径上的每一点的坐标都必须满足横坐标x>=纵坐标y，即路径必须在图中红色斜线(y=x+1)的下方，不能与红色斜线有交叉**。目标点（6,6）关于红色斜线对称的点为（5,7），则只要是从原点（0,0）走到（5,7）的路径必然会与红线有交叉，即不符合路径的要求。

（0,0）--> （6，6）的路径方案数：C_{12}^{6} （总方案）

（0,0）--> （5，7）的路径方案数：C_{12}^{5}   (不符合条件的方案)

故符合条件的方案数`C_{12}^{6} - C_{12}^{5}`
$$
由以上推广到n个0和n个1，得到公式:\\
C_{2n}^{n}-C_{2n}^{n-1}
\\=\frac{C_{2n}^{n}}{n+1}，即卡特兰数
$$
注：**这里结果依旧要对一个质数取模，所以可以用费马定理，快速幂求逆元**

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int mod=1e9+7;
typedef long long LL;

LL qmi(int a,int k,int p){
    LL res=1;
    while(k){
        if(k & 1) res=(LL)res*a%p;
        k=>>1;
        a=(LL)a*a%p;
    }
    return res;
}

//求卡特兰数   求组合数用基本公式法 和组合计数III一样 写法略有不同
LL Katelan(int n){
    int a=2*n,b=n;
    LL res=1;
    for(int i=a,j=1;j<=b;j++,i--){
        res=(LL)res*i%mod;
        res=(LL)res*qmi(j,mod-2,mod)%mod;
    }
    return (LL)res*qmi(n+1,mod-2,mod)%mod;//注意这里还有*(1/n+1)的逆元
}

int main(){
    int n;
    cin>>n;
    LL res=Katelan(n);
    cout<<res;
    return 0;
}

```



## （四）容斥原理

先不考虑重叠的情况，把包含于某内容中的所有对象的数目先计算出来，然后再把计数时重复计算的数目排斥出去，使得计算的结果既无遗漏又无重复，这种计数的方法称为**容斥原理**。

以S1,S2,S3三个集合为例，求出三个集合中包含的元素个数，可以通过韦恩图得到$S1∪S2∪S3 = S1+S2+S3- S1∩S2 - S1∩S3 - S2∩S3 + S1∩S2∩S3$。通过数学归纳法可以证明，对于求n个集合S1,S2,...,Sn集合中包含的元素个数，可以通过下面的公式来计算（**注意正负号交替**）：
$$
| S_{1}\cup S_{2} \dots \cup S_{m}  | = \sum_{i}  | S_{i}  | - \sum_{i,j} | S_{i} \cap S_{j}  |+ \sum_{i,j,k}  | S_{i}\cap S_{j} \cap S_{k}  |....+(-1)^{n-1}\sum  | S_{1}\cap S_{2} \cap S_{3}... \cap S_{m}  |........①
$$
计算总共的项数：利用组合数计算，每次从n个元素里面选i个进行交集计算，故**总项数**
$$
C_{n}^{1} + C_{n}^{2} + \dots + C_{n}^{n} = 2 ^ {n}
$$
即**时间复杂度为O($2^n$)**

接下来验证的一下上面①式中每个元素是不是只算了一次(具体证明略)：
$$
假设x\in S_{1} \cup S_{2} \dots \cup S_{n}，存在于k个集合之中，1\le k\le n
\\那么x被计算的次数为C_{k}^{1} - C_{k}^{2}+C_{k}^{3}-C_{k}^{4}+ \dots + (-1)^{k-1}C_{k}^{k}=1
$$

#### **AcWing 890.能被整除的数**

![img](assets\de1b878a6b19502e5e4855dd6538d4ea.png)

**实现思路：**记Si为1~n中能被pi整除的集合，根据容斥原理，所有数的个数为各个集合的并集，计算公式
$$
 | S_{1}\cup S_{2} \dots \cup S_{m}  | = \sum_{i}  | S_{i}  | - \sum_{i,j} | S_{i} \cap S_{j}  |+ \sum_{i,j,k}  | S_{i}\cap S_{j} \cap S_{k}  |....+(-1)^{n-1}\sum  | S_{1}\cap S_{2} \cap S_{3}... \cap S_{m}  |
$$

- 每个集合Si就对应能被质数pi整除的数

- 对于每个集合Si实际上并不需要知道含有哪些元素，**只需要知道各个集合中元素个数**，对于单个集合Si中**元素个数就是对应质数pi的倍数个数（1~n范围内）**，计算公式为
  $$
  |S_i|=\frac{n}{p_i}，下取整
  $$

- 对于**任意个集合交集中元素的个数**：每个质数pi对应一个集合Si，那么
  $$
  |S_i \cap S_j|=\frac{n}{p_i*p_j}，下取整，即交集就是p_i和p_j的公倍数的个数
  $$

- 表示**每个集合的状态（即选中几个集合的交集，）**：**m个质数，需要m个二进制位表示，共2^m-1种情况（至少选中一个集合），个数前面的符号为`(-1)^(n-1)`**。以m = 4为例，所以需要4个二进制位来表示每一个集合选中与不选的状态，若此时为1011，表示集合S1∩S3∩S4中元素的个数，同时集合个数为3，前面的符号为(-1)^(3-1)=1，即
  $$
  |S_1 \cap S_3 \cap S_4|=(-1)^{3-1}\frac{n}{p_1*p_3*p_4}
  $$
  **怎么取到一个数的每一个二进制位**：使用位运算（第一章）, 数`i`的第`j`位是否为1：`i>>m & 1`

  注：用二进制表示状态的小技巧非常常用，后面的状态压缩DP也用到了这个技巧，因此一定要掌握

```c++
#include <iostream>
using namespace std;

typedef long long LL;
const int N=20;
int p[N];//存储质数pi
int n,m;

int mian(){
    cin>>n>>m;
    for(int i=0;i<m;i++) cin>>p[i];
    
    int res=0;
    //枚举各个状态 0000..1到1111..1(m个质数) 至少选中一个集合
    for(int i=1;i<1<<m;i++){//1<<m，即1右移m位 有2^m-1种集合交的情况
        int t=1,s=0;//t表示质数乘积，s表示二进制中1的个数即哪几个质数相乘
        
        //枚举状态的每一位
        for(int j=0;j<m;j++){
            //得到每一个二进制位：位运算
            if(i>>m & 1){
                if((LL)t*p[j]>n){//乘积大于n 元素个数自然为0 跳出该轮循环
                    t=-1;//标记一下
                    break;
                }
                s++;//选中的集合个数
                t=(LL)t*p[j];//乘积
            }
        }
        //得到乘积后
        if(t==-1) continue;
        
        //偶数个集合的交为负 奇数为正
        if(s%2) res+=n/t;
        else res-+n/t;
    }
    cout<<res<<endl;
    return 0;
}

```





## （五）简单博弈论

**公平组合游戏ICG**

若一个游戏满足： 由两名玩家交替行动；

- 在游戏进程的任意时刻；
- 可以执行的合法行动与轮到哪名玩家无关；
- 不能行动的玩家判负；

则称该游戏为一个公平组合游戏。 NIM博弈属于公平组合游戏，但城建的棋类游戏，比如围棋，就不是公平组合游戏。因为围棋交战双方分别只能落黑子和白子，胜负判定也比较复杂，不满足条件2和条件3。

##### Nim游戏

给定`N`堆物品，第`i`堆物品有`Ai`个。两名玩家轮流行动，**每次可以任选一堆，取走任意多个物品，可把一堆取光，但不能不取**。取走最后一件物品者获胜。两人都采取最优策略，问先手是否必胜。

我们把这种游戏称为NIM博弈。把游戏过程中面临的状态称为局面。整局游戏第一个行动的称为先手，第二个行动的称为后手。若在某一局面下无论采取何种行动，都会输掉游戏，则称该局面必败。 所谓采取最优策略是指，若在某一局面下存在某种行动，使得行动后对面面临必败局面，则优先采取该行动。同时，这样的局面被称为必胜。我们讨论的博弈问题一般都只考虑理想情况，即两人均无失误，都采取最优策略行动时游戏的结果。 NIM博弈不存在平局，只有先手必胜和先手必败两种情况。 

- **必胜状态**，先手进行某一个操作，留给后手是一个必败状态时，对于先手来说是一个必胜状态。即**先手可以走到某一个必败状态**。
- **必败状态**，先手无论如何操作，留给后手都是一个必胜状态时，对于先手来说是一个必败状态。即**先手走不到任何一个必败状态。**

**结论**：假设n堆物品，数目分别为a1,a2,a3....an，如果`a1⊕a2⊕a3.......⊕an != 0`，则**先手必胜**；否则为0，先手必败（证明[AcWing 891. Nim游戏 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/562192/)）

1. 如果先手面对的局面是a1⊕a2⊕…⊕an≠0，那么先手总可以通过拿走某一堆若干个石子，将局面变成a1⊕a2⊕…⊕an=0。如此重复，最后一定是后手面临最终没有石子可拿的状态。先手必胜。
2. 如果先手面对的局面是a1⊕a2⊕…⊕an=0，那么无论先手怎么拿，都会将局面变成a1⊕a2⊕…⊕an≠0，那么后手总可以通过拿走某一堆若干个石子，将局面变成a1⊕a2⊕…⊕an=0。如此重复，最后一定是先手面临最终没有石子可拿的状态。先手必败。

#### AcWing891. Nim游戏

![img](assets\2066dbc357b4dfac5143d914c8c0f80e.png)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
int main(){
    int n;
    cin>>n;
    int res=0;
    while(n--){
        int a;
        cin>>a;
        res^=a;
    }
    if(res) puts("Yes");
    else puts("No");
    return 0;
}

```

#### AcWing892. 台阶-Nim游戏

![img](assets\ab00610e49a7760b4755eef7bb4695c6.png)

**实现思路：**

此时我们需要将**奇数台阶**看做一个经典的Nim游戏，**如果先手时奇数台阶上的值的异或值为0，则先手必败，反之必胜**

证明：

- 先手时，如果**奇数台阶异或非0**，根据经典Nim游戏，**先手总有一种方式使奇数台阶异或为0**，于是先手留了奇数台阶异或为0的状态给后手
- 于是轮到后手：
  - ①**当后手移动偶数台阶上的石子时**，先手只需将对手移动的石子继续移到下一个台阶，这样奇数台阶的石子相当于没变，于是留给后手的又是奇数台阶异或为0的状态
  - ②**当后手移动奇数台阶上的石子时**，留给先手的奇数台阶异或非0，根据经典Nim游戏，先手总能找出一种方案使奇数台阶异或为0

因此无论后手如何移动，**先手总能通过操作把奇数异或为0的情况留给后手**，当奇数台阶全为0时，只留下偶数台阶上有石子。
（核心就是：先手总是把奇数台阶异或为0的状态留给对面，即总是将必败态交给对面）

**因为偶数台阶上的石子要想移动到地面，必然需要经过偶数次移动**，又因为奇数台阶全0的情况是留给后手的，因此**先手总是可以将石先移动到地面**，当将最后一个（堆）石子移动到地面时，后手无法操作，即后手失败。

**故先手时奇数台阶上的值的异或值为非0，则先手必胜，反之必败！**

```c++
#include <iostream>
using namespace std;
int main(){
    int res=0;
    int n;
    cin>>n;
    for(int i=1;i<=n;i++){
        int x;
        cin>>x;
        if(i%2) res^=x;//选择奇数台阶进行异或
    }
    if(res) puts("Yes");//异或非0 必胜
    else puts("No");
}
```



**有向图游戏**

给定一个有向无环图，图中有一个唯一的起点，在起点上放有一枚棋子。两名玩家交替地把这枚棋子沿有向边进行移动，每次可以移动一步，无法移动者判负。该游戏被称为有向图游戏。 任何一个公平组合游戏都可以转化为有向图游戏。具体方法是，把每个局面看成图中的一个节点，并且从每个局面向沿着合法行动能够到达的下一个局面连有向边。



**Mex运算**

设S表示一个非负整数集合。定义mex(S)为求出**不属于集合S的最小非负整数**的运算，即： mex(S) = min{x}, x属于自然数，且x不属于S。eg：mex({1,2,3})=0，mex({0,1,2})=3



**SG函数**

在有向图游戏中，对于每个节点x，设从x出发共有k条有向边，分别到达节点y1, y2, …, yk，定义SG(x)为x的后继节点y1, y2, …, yk 的SG函数值构成的集合再执行mex(S)运算的结果，即： SG(x) = mex({SG(y1), SG(y2), …, SG(yk)}) 特别地，**整个有向图游戏G的SG函数值被定义为有向图游戏起点s的SG函数值，即SG(G) = SG(起点)**。且**图的终点（即不含出边的点）的SG函数值为0**.

**定理：**

- 有向图游戏的某个局面**必胜，当且仅当该局面对应节点的SG函数值不为0。** 
- 有向图游戏的某个局面**必败，当且仅当该局面对应节点的SG函数值等于0。**



**有向图游戏的和**

设G1，G2,····,Gm是m个有向图游戏.定义有向图游戏G,他的行动规则是任选某个有向图游戏Gi,并在Gi上行动一步.G被称为有向图游戏G1,G2,·····,Gm的和.
有向图游戏的和的SG函数值等于它包含的各个子游戏SG函数的异或和,即:**SG(G)=SG(G1)⊕SG(G2)⊕··⊕ SG(Gm)**



#### **AcWing 893.集合-Nim游戏**

![img](assets\fe19e3c9e971a47d9b272f97dc5f268d.png)

若有一堆的数量为10，取法为2，5，则他的状态图如下所示，就是每次选择取法不同，出现不一样的分支

<img src="assets\816ed959008c34c1923d2b8ed2324e49.png" alt="img" style="zoom:50%;" />

得到其SG函数的值，即SG(10)=1。同理计算其他石子堆的的SG值，**将所有堆SG值异或，若不为0，则先手必胜，若为0则先手必败**。（更具体解释[AcWing 893. 集合-Nim游戏 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/1404355/)）

**实现思路：**

- 设置取法数组s[]，所有石子堆共享操作

- 设置一个数组f，**`f[x]`记录图中某点x的SG值**，初始为-1表示x还未计算。**注意只要x的SG确定了那么就不会再改变**，因为针对同一个数x，无论在哪个石子堆，他就只能执行相同的操作，因为所有石子堆共享取法集合。比如x=5，取法{2,5}，那么在任何石子堆，x的出边只能为0或3，固定的；

- 这里使用一个**哈希表S存储当前点出边点的SG值，以此确定当前点的SG值**，可实现自动排序，和判断某个值是否出以此来完成mex操作（选出最小且没有出现的自然数）。**哈希表S必须定义为局部变量，每次递归都会使用新的哈希表**，所以一个图其他不与当前点相连的点的存在不会影响当前点SG的计算

  > 为什么哈希S不能开全局，只能作为局部变量？
  >
  > 对于集合-Nim，注意到0这个值可以被映射多次（如上图），这意味着有多个可能的值xi作为叶子节点（末尾节点），满足f[xi] = 0。而如果将S作为全局变量，则其中只能有一个值x’映射到0，由mex函数知其他的值都将映射到大于0的值（即f[x'] = 0，f[xi/x'] > 0，xi/x'代表集合x排除掉x’所剩值组成的集合），因而不能将S作为全局变量，防止不同点的相互影响。

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <set>
using namespace std;
const int N=110,M=10010;
int n,m;
int f[M],s[N];//f存储出现的某点的SG值，s存储取法数

//求各点的SG值
int sg(int x){
    if(f[x]!=-1) return f[x];//表示点x的SG值已经计算过 直接返回即可
    
    unordered_set<int> S;//哈希表 存储x能到的点的SG值，以此确定x的值 每次递归都是新的
    for(int i=0;i<m;i++){
        int op=s[i];//取一个操作数
        if(x>=op) S.insert(sg(x-op));//一条分支路径 直到终点
    }
    
    //mex操作 确定x的SG值
    for(int i=0;;i++)
        if(!S.count(i)) return f[x]=i;//选出最小的没有出现的自然数,赋给x的SG值
    
}

int mian(){
    cin>>m;
    for(int i=0;i<m;i++) cin>>s[i];
    memset(f,-1,sizeof f);//初始化为-1 表示还未计算
    cin>>n;
    int res=0;
    while(n--){
        int x;
        cin>>x;
        res^=sg(x);//各个堆的SG值异或，每个堆的SG=起点的SG值
    }
    if(res) puts("Yes");
    else puts("No");
    return 0;
}

```



#### AcWing894. 拆分-Nim游戏

![img](assets\53bf0c5492fb4c1c59806d11925fab96.png)

**实现思路**：首先可以看出必然会有解（有限的），因为每次放入的两堆规模各自都会比原规模小，各堆石子每次操作后呈递减趋势，最后必然会趋于0，比如一堆石子个数是1，取走以后，只会再放入两堆个数为0的石子。

相比于集合-Nim，这里的每一堆可以变成小于原来那堆的任意大小的两堆

即a[i]可以拆分成(b[i], b[j]),为了避免重复规定b[i]>= b[j],即: a[i]>b[i]>= b[j]

相当于一个局面拆分成了两个局面，由SG函数理论，多个独立局面的SG值，等于这些局面SC值的异或和。

因此需要存储的状态就是sg(b[2])^sg(b[])(与集合-Nim的唯一区别)

PS:因为这题中原堆拆分成的两个较小堆小于原堆即可，因此任意一个较小堆的拆分情况会被完全包含在较大堆中，因此S可以开全局。当然也可以在函数中定义。

```c++
#include <iostream>
#include <cstring>
#include <unordered_set>
using namespace std;
const int N=110;

int n;
int f[N];
unordered_set<int> S; //开成全局变量

//求sg的值
int sg(int x){
    if(f[x]!=-1) return f[x];//表示已经求过了 不需要再求
    
    //将一堆石子拆分为两堆更小的石子i j
    for(int i=0;i<x;i++)
        for(int j=0;j<=i;i++)//规定j不大于i，避免重复计算
            S.insert(sg(i)^sg(j));//由SG的函数理论，多个独立局面的SG值等于这些局面SG值的异或
    //mex操作 得到不存在的最小数
    for(int i=0;;i++)
        if(!S.count(i)) return f[x]=i;
}

int main()
{
    memset(f , -1 , sizeof f);

    cin >> n;
    int res = 0;
    while(n--)
    {
        int x;
        cin >> x;
        res ^= sg(x);
    }

    if(res) puts("Yes");
    else puts("No");
    return 0;
}

```



# 第五章 动态规划

![img](assets\d4de329c162984ef06a1018142a28f30.png)

DP问题，通常从2方面来思考：**状态表示**和**状态计算**

**状态表示**

从2方面考虑

1. 集合（某一个状态表示的是哪一种集合）

2. 属性（这个状态存的是集合的什么属性）

   一般属性有三种：集合的最大值，集合的最小值，集合中的元素个数

**状态计算** 

状态转移方程，即集合的划分。比如对 `f(i, j)`，考虑如何将其划分成若干个更小的子集合，而这些更小的子集合，又能划分为更更小的子集合。

集合的划分有2个原则：

- 不重：即不重复，某个元素不能既属于子集合A，又属于子集合B
- 不漏：即不漏掉任一元素，某个元素不能不属于任何一个子集合。

通常需要满足不漏原则，而不重不一定需要满足。

**动态规划的时间复杂度=状态数量*转移的计算量**



## 1.背包问题

**什么是背包问题**？

给定N个物品和一个容量为V的背包，每个物品有**体积**和**价值**两种属性，在一些限制条件下，将一些物品装入背包，使得在不超过背包体积的情况下，能够得到的最大价值。根据不同的限制条件，分为不同类型的背包问题。



### 0-1背包问题

给定`N`个物品，和一个容量为`V`的背包，每个物品有2个属性，分别是它的体积`v_i`(v for volume)，和它的价值`w_i `(w for weight)，每件物品只能使用一次（0-1背包的特点，每件物品要么用1次（放入背包），要么用0次（不放入背包）），问往背包里放入哪些物品，能够使得物品的总体积不超过背包的容量，且总价值最大。	

![img](assets\3162363661622e706e67)

`f(i, j)`可以分成两个更小的集合，一种是不包含第`i`个物品，一种是包含第`i`个物品

- 不包含第`i`个物品：就是从物品`1-i`中选择，但是不能包含第`i`个物品的最大价值，换句话就是从物品`1~i-1`中选择，总体积不超过`j`的最大价值，即`f(i - 1, j)`
- 包含第`i`个物品：就是从物品`1~i`中选择，但是必须包含第`i`个物品的最大价值，那么可以认为最开始直接把`i`塞进背包，此时背包的容量变成了`j - vi`，价值变成了`wi`，由于第`i`个物品已经装进背包了，那么从`1-i`选就变成了从`1-i-1`选了，因此此时的最大价值就是`f(i - 1, j - vi) + wi`

**`f(i, j)`取两种情况的最大值，因此`f(i, j)= max(f(i - 1, j), f(i - 1, j - vi) + wi)`**

#### **AcWing 2. 01背包问题**

![img](assets\c59d47f5fc5bbd5d4c8dc60664ea3c4d.png)

**实现思路：**求`f(i,j)`，`i`从0开始枚举到n件物品，再用`j`从0开始枚举到最大体积m，由于包含`i`的集合可能不存在，因此先计算不包含`i`的集合，即`f(i,j)=f(i-1,j)`，若当前的状态可以划分包含`i`的状态，即`j>=v[i]`,那么就计算当前枚举的`f(i,j)`最终值，即`max(f((i-1),j),f(i-1,j-v[i])+w[i]))`，当全部枚举结束后，计算的就是`f[n][m]`，即前n个物品中总体积不超过m的最大价值。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
int n,m;
int v[N],w[N],f[N][N];//体积、价值、最大价值

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for(int i=1;i<=n;i++)//f[0][j]都默认为0
        for(int j=1;j<=m;j++){//f[i][0]都默认为0
            f[i][j]=f[i-1][j];//不包含物品i的情况
            if(j>=v[i]) f[i][j]=max(f[i][j],f[i-1][j-v[i]]+w[i]);//包含物品i，直接先放进去
        }
    cout<<f[n][m]<<endl;
    return 0;
}


```

#### 滚动数组优化为一维

将状态`f[i][j]`优化到一维`f[j]`，实际上只需要做一个**等价变形**。

为什么可以这样变形呢？我们定义的状态`f[i][j]`可以求得任意合法的`i`与`j`最优解，即放前i个物品在体积为j时的最大价值（很多个状态都可以得到），但题目只需要求得**最终状态**`f[n][m]`（只要一个状态，不用求那么多状态），因此我们**只需要一维的空间来更新状态**。

（1）状态`f[j]`定义：N件物品，背包容量`j`下的最优解(最大价值)。

（2）注意枚举背包容量`j`必须从`m`开始，即**逆序**遍历处理不同体积。

**为什么一维情况下枚举背包容量需要逆序？**

在二维情况下，状态`f[i][j]`是由上一轮`i - 1`的状态得来的，`f[i][j]`与`f[i - 1][j]`是独立的。而优化到一维后，如果我们还是正序，则有`f[较小体积]`更新到`f[较大体积]`，则**有可能本应该用第`i-1`轮的状态却用的是第`i`轮的状态。**

**再具体来说：**二维削减为一维，`f[j]=max(f[j],f[j-v[i]]+wi)`

`f[j]`从`f[j - vi]`转移过来，那么这里的`f[j - w]`是指`f[i - 1][j - w]`还是`f[i][j - w]`就尤为重要

- 如果是**正序循环**，显然`j-vi<j`，**由于j是递增的**，那么**在一轮i循环**中`f[j-vi]`必然会在`f[j]`之前得到，则算到最后`f[m]=max(f[m],f[m-vi]+wi)`可能这时候的`f[m-vi]`在本轮循环中**已经更新修改过**（比如在开头的`j`处）。**就出现了某次更新`f[j]`的时候就用到了本轮`i`中的`f[j-vi]`（就是二维中的`f[i][j-vi]`，而不是应该用的`f[i-1][j-vi]`，错误），不是用的上一轮`i-1`中的`f[j-vi]`，这就不对了!!!**（这里正序实际是完全背包问题的做法）
- 如果是**逆序循环**，就可以完美解决这个问题，**j是递减的**， 计算`f[j]`的时候，**用到的`f[j - vi]`必然还没在本轮更新（还是`i-1`时候的，旧的好）**

- 简单来说，**一维情况正序更新状态f[j]需要用到前面计算的状态已经被「污染」，逆序则不会有这样的问题。**


一维状态转移方程为：`f[j] = max(f[j], f[j - v[i]] + w[i] )`

不理解？传送门：http://t.csdnimg.cn/Ue73u

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
int n,m;
int v[N],w[N],f[N];//体积、价值、最大价值

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for(int i=1;i<=n;i++)
        //j逆序，到v[i]为止（小于vi就没意义，装不下vi）
        for(int j=m;j>=v[i];j--){
            //f[j]=f[j];//不包含物品i的情况 改为一维后恒等式 略去
            f[j]=max(f[j],f[j-v[i]]+w[i]);//包含物品i 逆序使f[j-v[i]]等效于二维的f[i-1][j-v[i]]
        }
    cout<<f[m]<<endl;
    return 0;
}

```



### 完全背包问题

相比0-1背包问题，完全背包问题的**各个物品是无限个**的，即放入背包的物品i可以不限数量

#### **AcWing 3. 完全背包问题**

![img](assets\bb7e4edb3b43f3626a023aaa73cc786e.png)

**实现思路：**和0-1背包问题的区别在状态计算中的集合划分，不是只有0和1，而是可以选k个

![img](assets\706e67)

**朴素做法**：与01背包思路相同，只是在集合划分上有所区别，以`f[i,j]`为例，对其进行下一步划分，考虑以取`k`个`i`物品划分集合，若`k=0`，则相当于`f[i-1,j]`；若`k`不等于0，则采取01背包类似的办法，先确定取`k`个物品`i`，不影响最终选法的求解，即求`f[i-1,j-k*v[i]]`,再加上`k*w[i]`，即`f[i-1,j-k*v[i]]+k*w[i]`,不难发现k=0情况可以与之合并，最终就是取从`0`枚举到`k`，最终状态转移方程为`f[i][j]=max(f[i][j],f[i-1][j-k*v[i]]+k*w[i])`的最大值，k的最大值可以通过`j>=k*v[i]`求解。**有三重for循环，时间复杂度最差为O(n*m^2)**

注意：这里`max(f[i][j],f[i-1][j-k*v[i]]+k*w[i])`，而不是类似0-1背包那样取`max(f[i-1][j],f[i-1][j-k*v[i]]+k*w[i])`。因为k=0时，即物品`i`不取的情况，完全背包方程就为`max(f[i][j],f[i-1][j])`，实质上就涵盖了`f[i-1][j]`的情况

```c++
#include <iostream>
#include <alforithm>
using namespace std;
const int N=1010;
int w[N],v[N],f[N][N];
int n,m;


int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            for(int k=0;k*v[i]<=j;k++){
                f[i][j]=max(f[i][j],f[i-1][j-k*v[i]]+k*w[i]);
            }
    cout<<f[n][m]<<endl;
    return 0;
}
```

**二维优化版：**改为二重循环，降低时间复杂度

像0-1背包那样考虑分成两种情况看待，

第一种情况：从`i`物品一个都不取开始；

第二种情况：从至少取一份`i`物品开始，即`j-v`

`f[i , j ] = max( f[i-1,j] , f[i-1,j-v]+w ,  f[i-1,j-2*v]+2*w , f[i-1,j-3*v]+3*w , .....)`
`f[i , j-v]= max(            f[i-1,j-v]   ,  f[i-1,j-2*v]+ w ,  f[i-1,j-3*v]+2*w , .....)`
观察上面两式，括号中对应部分只相差一个w，可得出如下递推关系： 
                        `f[i][j]=max( f[i-1][j],f[i,j-v]+w ) `

**所以可以去掉k，即去掉第三重循环**

```c++
#include <iostream>
#include <alforithm>
using namespace std;
const int N=1010;
int w[N],v[N],f[N][N];
int n,m;


int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++){
            f[i][j]=f[i-1][j];//不取物品i
            if(j>=v[i]) f[i][j]=max(f[i][j],f[i][j-v[i]]+w[i]);//至少取一份物品i
        }
    cout<<f[n][m]<<endl;
    return 0;
}
```

**滚动数组优化**

观察可以发现可0-1背包的代码很像，所以可以像0-1背包那样用滚动数组优化。

**区别在于：**第二部分是`i-1`，还是`i`，即需要的值是上一轮的`i-1`还是本轮的`i`

`f[i][j] = max(f[i][j],f[i-1][j-v[i]]+w[i]);`//01背包 需要`i-1`轮的值来更新

`f[i][j] = max(f[i][j],f[i][j-v[i]]+w[i]);`//完全背包问题  需要`i`轮的值来更新

**相比0-1背包，进行滚动优化区别**：`j`是**正序**遍历处理了（而0-1背包的`j`是逆序）

详细一点说明：http://t.csdnimg.cn/633A5

```c++
#include <iostream>
#include <alforithm>
using namespace std;
const int N=1010;
int w[N],v[N],f[N];
int n,m;


int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for(int i=1;i<=n;i++)
        for(int j=v[i];j<=m;j++)//正序遍历j
            f[j]=max(f[j],f[j-v[i]]+w[i]);
    
    cout<<f[m]<<endl;
    return 0;
}
```



### 多重背包问题

每件物品的个数是不同的，比如，每件物品的个数是`si`个。

相比完全背包问题，只是每个物品的个数有了上限，不再是无限

![img](assets\c96d21494dbddac613bf49211a489510.png)

![Image](assets\132612e706e67)

**朴素版本：**和完全背包问题基本一样，只是k多了个上限限制，用数组s[]表示某个物品的上限。时间复杂度为**O(NVS)，会超时**

```c++
#include <iostream>
#include <alforithm>
using namespace std;
const int N=1010;
int w[N],v[N],f[N][N];
int s[N];//上限
int n,v;


int main(){
    cin>>n>>v;
    for(int i=1;i<=n;i++) cin>>v[i]>>w[i]>>s[i];
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=v;j++)
            for(int k=0;k<=s[i] && k*v[i]<=j;k++){//多一个上限的判断
                f[i][j]=max(f[i][j],f[i-1][j-k*v[i]]+k*w[i]);
            }
    cout<<f[n][v]<<endl;
    return 0;
}
```

**二进制优化：**分堆，最后退化为0-1背包问题

- 所谓的二进制优化，我们知道**任意一个实数可以由二进制数来表示，也就是可以由$2^0$~$2^k$其中一项或几项的和拼凑得到**。

  比如1,2,4,8,16可以组合表示出0~31中间的任何数，可推导得到**$2^1,2^2.....2^k$可以组合表示$0$~$2^{k+1}-1$中的任何数，即用$log(2^k+1)$个数就可以表示$2^{k+1}$个数**，妙啊！

  那么对于`0~S[i]`，本来循环枚举`S[i]`次，若使用二进制优化，可以只枚举`logS[i]`次就可以表示出`0~S[i]`范围的所有数，**枚举的时间复杂度O(S)降低到S(logS)**

- 然后对应到多重背包问题：对于物品`i`限制上限`S[i]`个，我们对其取多少个进行**分堆**，每堆的个数为2的倍数，即进行二进制优化，**每次就按堆来枚举询问**，是否达到最优解，再通俗点说就是**原来物品`i`按每次一个一个询问是否最优，转化为每次一堆一堆（每一堆最多只能用一次，即转化为0-1背包问题了）询问是否最优**，效果一样的，因为DP就是找到最优就行。

- 怎么分堆（怎么转为每堆尽量用2的倍数表示）？

  比如`s[i]=200`，那么分堆1,2,4,8.....64，此时可表示0~127（若分到128，就表示0~255，超出200了），还剩下一堆用`s[i]-127=73`，这样就可以表示0~200中的数。也就是说分堆的时候，先分为`k`堆，且`2^k-1<s[i]`，但`2^(k+1)-1>s[i]`，然后再用一堆放`s[i]-(2^k-1)`，这样就分为`k+1`堆，原来`s[i]`个数转化为`k+1`个数表示。

**经过二进制优化后，时间复杂度为O(NVlogS)**，运行不会报超时

**注意**：这里分堆后，**直接覆盖体积数组v[]和价值数组**，因为每次询问就是一堆一堆了,`v[i]=堆中个数k*v`，`w[i]=堆中个数k*w`

更详细说明：[AcWing 5. 二进制优化，它为什么正确，为什么合理，凭什么可以这样分？？ - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/1270046/)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=25000,M=2010;
int n,m;
int w[N],v[N],f[N];
int n,m;

int mian(){
    cin>>n>>m;
    int cnt=0;//记录堆号 第几堆
    for(int i=1;i<=n;i++){
        int v,w,s;//当前物品i的体积、价值、上限个数
        cin>>v>>w>>s;
        
        int k=1;//记录每堆的个数 递增从1 2 4 8 到剩余的个数不再满足2的倍数 重开一个堆放
        while(s>=k){
            cnt++;
            v[cnt]=k*v;//堆的体积
            w[cnt]=k*w;//堆的价值
            s-=k;//剩余个数
            k*=2;//递增分下一堆
        }
        if(s>0){//按2的倍数分完还有剩 再开一个堆
            cnt++;
            v[cnt]=s*v;//堆的体积
            w[cnt]=s*w;//堆的价值
        }
    }
    
    //分好了 各堆只能用一次 0-1背包问题
    n=cnt;
    for(int i=1;i<=n;i++)
        for(int j=m;j>=v[i];j--)//注意逆序
            f[j]=max(f[j],f[j-v[i]]+w[i]);
    cout<<f[m];
    return 0;
}

```



### 分组背包问题

有 N 组物品，**每一组中有若干个物品**，**每一组中至多选择一个**。

分组背包问题的思考方式和前面的类似。不同的地方仅仅在于状态转移。

![img](assets\efa87ec0989a7d51af17b4e27b434c3d.png)

分组背包问题的思考方式和前面的类似。不同的地方仅仅在于状态转移。

01背包的状态转移，是枚举第`i`个物品选或者不选；

完全背包和多重背包，是枚举第`i`个物品，选`0,1,2,3,4,....` 个，无限个或有上限个

**而分组背包，枚举的是第`i`个分组，选哪一个，或者一个都不选**

![img](assets\30382e706e67)

- 这里的体积数组`v`和价值数组`w`就要开成二维，表示某一组的某一个物品

- 与01背包思路一致，集合划分为不包含`i`组，包含`i`组第1个物品，包含i组第2个物品，...包含`i`组第`k`个物品（`k`表示第`i`组的物品数量），...，包含第`i`组最后一个物品。因此若不包含第`i`组，则`f(i,j)=f(i-1,j)`；若包含第`i`组第`k`个物品，则计算方法类似01背包（只是多了一重循环从`i`组里面选第`k`个物品），先除去第`i`组的第`k`个物品再进行计算的取法不变
- 分组背包的状态转移方程为：

`f[i][j]=max(f[i−1][j],f[i−1][j−v[i][k]]+w[i][k])`， `1<k<s[i]`。其中 `v[i,k] `表示第` i` 组中的第 `k `个物品的体积，`w [ i , k ]` 同理。同样可以优化为一维：`f[j]=max(f[j],f[j−v[i][k]]+w[i][k])`，主要这里更新需要上一组的（和完全背包一样），`j`要逆序

**注意**：`j`逆序枚举的最小值是1，不是0-1背包那样的`v[i]`，因为`i`组里面的物品各自的体积都无法提前判断，不知道最小值

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N =110;
int v[N][N],w[N][N],s[N],f[N];//有些开成二维数组,s表示第i组中的物品数量
int n,m;

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>s[i];
        for(int j=1;j<=s[i];j++)
            cin>>v[i][j]>>w[i][j];
    }
    for(int i=1;i<=n;i++)
        for(int j=m;j>=1;j--)//逆序枚举体积 这里就和01背包有区别 枚举最小到1
            for(int k=1;k<=s[i];k++)
                if(j>=v[i][k]) //满足体积还足够放i组第k个
                    f[j]=max(f[j],f[j-v[i][k]]+w[i][k]);
    cout<<f[m]<<endl;
    return 0;
}
```



## 2.线性DP

状态转移方程呈现出一种线性的递推形式的DP，我们将其称为线性DP。

### （一）数字三角形

#### AcWing 898. 数字三角形

![img](assets\f8d735153713905b41b0e0d9b4a394c5.png)

**实现思路：**

- 对这个三角形中的数字进行编号，状态表示依然可以用二维表示，即`f(i,j)`，`i`表示横坐标（横线），`j`表示纵坐标（斜线）

<img src="assets\86c68268b08ecc49a7dfa494c0fc2c6f.png" alt="img" style="zoom:50%;" />

![img](assets\30306332622e706e67)

- 用`f(i,j)`表示到点`(i,j)`的路径最大数字之和。对集合进行划分，**到达某点`(i,j)`只可能经过左上方的点`(i-1,j-1)`或右上方的点`(i-1,j)`**。用`a[i][j]`表示当前点的数值
- 故可得状态转移方程：`f[i][j]=max(f[i-1][j-1]+a[i][j],f[i-1][j]+a[i][j])`

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=510,INF=1e9;
int n;
int f[N][N],a[N][N];

int mian(){
    cin>>n;
    for(int i=0;i<=n;i++)
        for(int j=0;j<=i+1;j++)//注意这里最大下标为i+1 因为在每一行的最后一个数的右上方没数但也会用到
            a[i][j]=-INF;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=i;i++)
            cin>>a[i][j];
    f[1][1]=a[1][1];//初始化到点（1,1）的最大值之和
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=i;j++)
            f[i][j]=max(f[i-1][j-1]+a[i][j],f[i-1][j]+a[i][j]);//左上 右上 的最大值
    
    int res=-INF;
    for(int i=1;i<=n;i++)//遍历最后一行选出最大值
        res=max(res,f[n][i]);
    cout<<res;
    
    return 0;
}

```

这道题还可以从下往上递推，考虑`f[i][j]`来自左下方和来自右下方两种情况，这样就不需要处理边界问题，而且最后的结果一定集中在`f[1][1]`中。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=510,INF=1e9;
int n;
int f[N][N],a[N][N];

int mian(){
    int n;
    cin >> n;
    for (int i= 1; i <= n; i++)
        for (int j = 1; j <= i; j++)
            cin >> a[i][j];
            
    for (int i = 1; i <= n; i++) f[n][i] = a[n][i];
    
    for (int i = n - 1; i >= 1; i--)
        for (int j = 1; j <= i; j++)
            f[i][j] = max(f[i + 1][j], f[i + 1][j + 1]) + a[i][j];
    
    cout << f[1][1] << endl;
    return 0;
}

```





### （二）最长上升子序列(LIS)

#### AcWing 895. 最长上升子序列

![img](assets\821ad1f9cb6642b35c6e2d7de80a0a98.png)

**实现思路**：

<img src="assets\633382e706e67" alt="img" style="zoom:80%;" />

- 一维数组`f[i]`表示以第`i`个数为结尾的最长递增子序列的长度。
- 状态划分：选定`i`为结尾的递增子序列，则再从`[0,i-1]`中筛选出倒数第二个位置的数，使递增子序列的长度最大。注意这个倒数第二个位置的**数必须满足`a[j]<a[i]`，这样才能保证递增序列**。
- 状态转移方程为`f[i]=max(f[i],f[j]+1);`
- 时间复杂度为O(n^2)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
int f[N],a[N];
int n;
int mian(){
    cin>>n;
    for(int i=1;i<=n;i++)
        cin>>a[i],f[i]=1;//f[i]初始为1即仅包括自身
    for(int i=1;i<=n;i++)
        for(int j=1;j<i;j++)
            if(a[j]<a[i]) f[i]=max(f[i],f[j]+1);
    int res=0;
    for(int i=1;i<=n;i++) res=max(res,f[i]);//得到最大值
    cout<<res<<endl;
    return 0;
}

```



#### 二分优化

**当数据范围扩大时，O(n^2)的时间复杂度已经不能满足要求时用二分的方法可以优化到O(nlogn)**

![img](assets\6de22093ce334bad5dc3adf76dedd5c5.png)

**实现思路**：

- 首先在上述解法的基础上，假如存在一个序列3 1 2 5，以3结尾的上升子序列长度为1，以1为结尾的上升子序列长度也为1，这是**两个长度一样的上升子序列**（伏笔：**结尾元素1<3**）。在继续向后遍历查找时，看3这个序列，当出现一个比3大的数时，以3结尾的上升子序列就会更新，比如遍历到5了，那么上升序列变为3 5；同时注意到这个5一定会加入到以1结尾的上升序列中（因为1<3，那么1<5的），那么含有1的上升序列长度一定是>=2的，因为中间可能存在<3但>1的数（比如这里就有2，序列长度就更新为3）。可以看出存在3的这个序列就不需要枚举了，**因为存在1的序列往后遍历的长度是一定大于你这个存在3的序列的（前提是以1结尾和以3结尾的上升序列长度相等），那我找最长的时候怎么都轮不到包含3的序列头上**，那我一开始在1和3结尾的序列之后直接舍弃枚举包含3的序列了（去掉冗余）。
- 在以上的分析得到：**当存在两个上升序列长度相同时，结尾数更大的序列可以舍去不再枚举，所以每次就干脆选出相同长度结尾元素最小的序列继续操作**
- 那么**状态表示**更改为：`f[i]`**表示长度为`i+1`(因为下标从0开始)的最长上升子序列，末尾最小的数字**。(**所有长度为`i+1`**的最长上升子序列所有结尾中，结尾最小的数) 即长度为`i`的子序列末尾最小元素是什么。
- **状态计算**：对于每一个数`w[i]`, **如果大于`f[cnt-1]`**(下标从`0`开始，`cnt`长度的最长上升子序列，末尾最小的数字)，那就将这个数`w[i]`添加到当前序列末尾，使得最长上升序列长度`+1`(`cnt++`)，当前末尾最小元素变为`w[i]`。 **若`w[i]`小于等于`f[cnt-1]`,**说明不会更新当前序列的长度，**但之前某个序列末尾的最小元素要发生变化**，找到第一个 **大于或等于(不能直接写大于，要保证单增)** `w[i]`的数的位置`k`，将这个数`w[i]`放在`k`的位置（**其实就是找到`w[i]`适合存在的位置，这里就使用二分查找，更新保证长度为k+1的序列的末尾元素为最小`w[i]`，即`f[k]=w[i]`**）。

其他参考说明：

[AcWing 896. 最长上升子序列 II - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/4065993/)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
int w[N],f[N];
int n,cnt;//cnt记录最大上升序列长度

int mian(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>w[i];
    
    f[cnt++]=w[0];//初始就放一个w[0]
    
    for(int i=1;i<=n;i++){
        if(w[i]>f[cnt-1]) f[cnt++]=w[i];//w[i]大于当前上升序列末尾的数 则末尾加入
        else{//否则 二分查找当前上升序列中合适的位置插入
            int l=0,r=cnt-1;
            while(l<r){
                int mid=(l+r)>>2;
                if(f[mid]>=w[i]) r=mid;
                else l=mid+1;
            }
            //找到合适位置了 此时长度为r+1的序列末尾最小元素即为w[i]
            f[r]=w[i]
        }
    }
    cout<<cnt<<endl;
    return 0;
}
```





### （三）最长公共子序列(LCS)

#### **AcWing 897.最长公共子序列**

![img](assets\a2953a9f1f6b2a585a05dfffe1cad276.png)

**实现思路**：

![img](assets\42e706e67)

- `f(i,j)`表示第一个序列的前`i`个字母和第二序列前`j`个字母**最长的公共子序列长度**
- 状态可划分为4中情况：`a[i]`表示第一个序列中第`i`个字符，`b[j]`表示第二个子序列中第`j`个字符
  - **00**：表示最长公共子序列中一定不包含字符`a[i]`和`b[j]`，用`f[i-1][j-1]`表示
  - **01**：表示最长公共子序列中**一定不包含字符`a[i]`，一定包含`b[j]`**。不能用`f[i-1][j]`表示（**不是等价的**），因为`f[i-1][j]`表示的是该公共子序列一定不包含a[i]，**但b[j]不一定，可能包含也可能不包含**。**故`f[i-1][j]`是包含01这种情况的**。但是由于求的是最大子序列的长度（而不是具体元素），**所以求解的时候可以用`f[i-1][j]`来求解**
  - **10**：表示最长公共子序列中一定包含字符`a[i]`，一定不包含`b[j]`。用`f[i][j-1]`求解，但含义不等价，同上。
  - **11**：表示最长公共子序列中一定包含字符`a[i]`和`b[j]`，用`f[i-1][j-1]+1`表示，但注意需要满足`a[i] = b[j]`才行，因为公共子序列，既然包含`a[i]`、`b[i]`，那么两者必然相等才行
- 注意：**00的情况实质上已经被包含在01、10两种情况之中**，所以可以省略，故只需求下面三种状态

更详细理解：

[AcWing 897. 最长公共子序列（思路超清晰） - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/2262673/)

[AcWing 897. 最长公共子序列 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/534717/)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
int n,m;
char a[N],b[N];
int f[N][N];

int mian(){
    cin>>n>>m;
    cin>>a+1>>b+1;//下标从1开始 往后移一位输入
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++){
            f[i][j]=max(f[i-1][j],f[i][j-1]);//前两种情况01 10 包含00的情况
            if(a[i]==b[j]) f[i][j]=max(f[i][j],f[i-1][j-1]+1);//最后一种情况11
        }
    cout<<f[n][m];
    return 0;
}


```



### （四）编辑距离

#### **AcWing 902.最短编辑距离**

![img](assets\3c0739af5e66450225107faedfb9a067.png)

**实现思路：**

![img](assets\3365332e706e67)

- `f(i,j)`表示，集合为所有将第一个字符串前`i`个字符变为第二个字符串前`j`个字符的方式的最少操作数量
- 集合划分：以第一个字符串`i`处可能进行的三种不同操作后转化为第二个字符串。
  - **删去**第`i`个字符，即前`i-1`个字符已经与第二个字符串的前`j`个字符相同，因此只需要在上一个状态加上删去操作即可，即`f(i,j)=f(i-1,j)+1`；
  - 第`i`个字符后面**增加**一个字符，即第一个字符串前`i`个字符已经与第二个字符串的前`j-1`个字符相同，需要在第一个字符串的末尾加上一个字符，因此只需要在上一个状态上加上插入操作即可，即`f(i,j)=f(i,j-1)+1`；
  - **修改**第`i`个字符，即前`i-1`个字符已经与第二个字符串的前`j-1`个字符相同，再比较第i个字符是否与第j个字符相同，若相同就不用操作，若不同则需要增加一次修改操作,即`f(i,j)=f(i-1,j-1)+0 or 1`。
- 最终`f(i,j)`取三者最小值

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;
char a[N],b[N];
int f[N][N];
int n,m;

int main(){
    //均从下标1开始
	scanf("%d%s",&n,a+1);
    scanf("%d%s",&m,b+1);
    
    for(int i=0;i<=n;i++) f[i][0]=i;//第二个字符串为空 则一直删除
    for(int i=0;i<=m;i++) f[0][i]=i;//第一个字符串为空 则一直增加
    
    for(int i=;i<=n;i++)
        for(int j=1;j<=m;j++){
            f[i][j]=min(f[i-1][j]+1,f[i][j-1]+1);//增 删
            
            if(a[i]==b[j]) f[i][j]=min(f[i][j],f[i-1][j-1]);//最后一个字符相等 不需要改
            else f[i][j]=min(f[i][j],f[i-1][j-1]+1);//最后一个不等 需要改
        }
    cout<<f[n][m];
    return 0;
}
```



#### **AcWing 899.编辑距离**

![img](assets\50c5f1bdc782822cb339858dc5598543.png)

**实现思路：**与上题思路一致，不过在读入时有所区别，该题需要读入n个字符串，m次问询，因此读入n个第一个字符串，然后在每次问询中读入第二个字符串，计算n个第一个字符串要变化到第二个字符串的次数，统计在规定次数内的第一个字符串有几个。

```c++
#include <iostream>
#include <algorithm>
#include <string.h>
using namespace std;
const int N = 15, M = 1010;
int n, m;
int f[N][N];
char str[M][N];//读入多个字符串

int edit_distance(char a[],char b[]){
    int al=strlen(a+1),bl=strlen(b+1);
    for(int i=0;i<=al;i++) f[i][0]=i;
    for(int i=0;i<=bl;i++) f[0][i]=i;
    
    for(int i=1;i<=al;i++)
        for(int j=1;j<=bl;j++){
            f[i][j]=min(f[i][j-1]+1,f[i-1][j]+1);
            if(a[i]==b[j]) f[i][j]=min(f[i][j],f[i-1][j-1]);
            else f[i][j]=min(f[i][j],f[i-1][j-1]+1);
        }
    return f[al][bl];
}

int main()
{
    scanf("%d%d", &n,&m);
    for (int i=0;i<n;i++) scanf("%s",str[i]+1);
    while (m--)
    {
        char s[N];
        int limit;
        scanf("%s%d",s+1,&limit);
        int res=0;
        for (int i=0;i<n;i++)
            if (edit_distance(str[i],s) <= limit)
                res ++ ;
        printf("%d\n", res);
    }
    return 0;
}

```



## 3.区间DP

#### AcWing 282.  石子合并

![img](assets\c49614a1eb1dc4af19f1b42d82b5f6c3.png)

**实现思路：**

![img](assets\5322e706e67)

- `f(i,j)`表示将第`i`堆到第`j`堆石子合并为一堆时的最小代价
- **状态划分**：选一个分割点`k`，将`i~k`，`k+1~j`这两个堆（两个区间）的石子合并，然后加上两个区间堆的总合并代价（采用**前缀和**计算区间`i`到`j`的值，`s[j]-s[i-1]`）。
- 初始从枚举区间长度开始（即石子堆数，实际作为不同的区间划分情况），区间长度`len`从`2`到`n`枚举(从2开始是因为，若区间长度只有1的话，没必要合并了)
- 然后枚举左端点`i`，从`1`到`i+len-1`
- `k`从左端点`i`开始枚举，比如`k=i+1`时，区间被分割为`(i,i+1)，(i+2,i+len-1)`，左边区间就一堆，右边区间`len-1`堆
- 设`l=i,r=i+len-1`，状态转移方程：`f[l][r]=max(f[l][r],f[l][k]+f[k+1][r]+s[r]-s[l-1])`

更详细说明：[AcWing 282. 石子合并 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/5658102/)

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=310;
int n;
int s[N];//求区间前缀和 
int f[N][N];
int mian(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>s[i];
        s[i]+=s[i-1];//同时计算前缀和
    }
    
    for(int len=2;len<=n;i++){//枚举区间长度（石子堆数）
        for(int i=1;i+len-1<=n;i++){//枚举左端点
            int l=i,r=i+len-1;
            f[l][r]=0x3f3f3f3f;//代价先初始化为无穷大
            for(int k=l;k<=r;k++)
                f[l][r]=min(f[l][r],f[l][k]+f[k+1][r]+s[r]-s[l-1]);
        }
    }
    
    cout<<f[1][n]<<endl;
    
    return 0;
}

```



## 4.计数类DP

#### AcWing 900. 整数划分

![img](assets\57fb55efa0b35e0ad35b019c91cba485.png)

**实现思路**：本题求的是方案个数，而不要求方案顺序，即4=1+1+2 和4=1+2+1是一样的

（1）**方案一：转化为完全背包做法**。将正整数n看做是背包容量，而1~n之间的数看做是物品，且各个物品的数量是无限的，至此转化为完全背包问题。

![img](assets\362e706e67)

- `f(i,j)`表示从前`i`个数字（物品）中选择，之和恰好是`j`（体积）的方案个数

- 以第`i`个数字选择了几次（物品`i`放了几个）做集合划分。若只选0个`i`，那么前`i-1`数的选择之和已经满足`j`，故为`f[i-1][j]`；若第`i`个数字选择了`k`次，那么前`i-1`个数的选择之和为`j-k*v[i]`，故`f[i-1][j-v[i]]`

- 类似完全背包问题的分析与优化：

  `f[i][j] = f[i - 1][j] + f[i - 1][j - i] + f[i - 1][j - 2i] + .... + f[i - 1][j - k*i]`

  `f[i][j - i] =           f[i - 1][j - i] + f[i - 1][j - 2i] + .... + f[i - 1][j - k*i]`

- 所以：

  **状态转移方程**：`f[i][j] = f[i - 1][j] + f[i][j - i]`

  优化至一维

  `f[j] = f[j] + f[j - i]`表示和为`j`的方案数量

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010,mod=1e9+7;//最后结果要取模
int n,f[N];
int main(){
    cin>>n;
    f[0]=1;//和为0，那就一种选法
    for(int i=1;i<=n;i++)
        for(int j=i;j<=n;j++)//j正序，且从i开始 省去判断j-i是否大于0
            f[j]=(f[j]+f[j-i])%mod;
    
    cout<<f[n]<<endl;
    return 0;
}

```

**（2）方案二**：

![img](assets\32e706e67)



- 用`f[i][j]`表示，所有总和是`i`，并且恰好表示成`j`个数之和的方案的数量。

- 集合划分，能够分为如下两类
  - 方案中最小值是1的所有方案，这时候去掉一个1，此时和变成了`i - 1` ，个数变成了`j - 1` ，即`f[i - 1][j - 1]`
  - 方案中最小值大于1的所有方案，此时将`j`个数都减去1，此时和变成了`i - j`(`j`个数每个数都`-1`，共`-j`)，个数还是`j`，即`f[i - j][j]`
- 最终状态转移方程为：`f[i][j] = f[i - 1][j-1] + f[i-j][j]`
- 结果输出应为`f[n][1]` + `f[n][2]` + `f[n][3]` + … + `f[n][n]`

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010,mod=1e9+7;
int n,f[N][N];

int main(){
    cin>>n;
    f[0][0]=1;//和为0 0个数表示的方案为1
    
    for(int i=1;i<=n;i++)
        for(int j=1;j<=i;j++)//和为i 最多表示为i个数的和(即i个1的和)
            f[i][j]=(f[i-1][j-1]+f[i-j][j])%mod;
    int res=0;
    for(int i=1;i<=n;i++)
        res+=f[n][i];
    cout<<res<<endl;
    
    return 0;
}

```



## 5.数位统计DP

#### AcWing 338. 计数问题

![img](assets\870150e353819c2443d35bcae67cead3.png)

**实现思路**：

- 定义函数：`count(n,x)`，其表示在`1`到`n`中，`x`出现的次数（`x`是0-9）

  那么，可以**用类似前缀和的思想**，来求解`a`到`b`中，`x`出现的次数：`count(b,x) - count(a-1,x)`

- 以`x = 1`为例，如何计算`count(n, 1)`：分情况讨论。比如`n`是个7位的数字 `abcdefg`，**我们可以分别求出`1`在每一位上出现的次数，然后做一个累加即可**。比如求`1`在第`4`位上出现的次数，求有多少个形如`xxx1yyy`的数在`1`到`abcdefg`之间。分两种大情况

  - 若`xxx<abc`，`xxx = 000 ~ abc - 1`, 中间`d=1`，`yyy = 000 ~ 999`，一共有`abc * 1000`(即左右两边数的大小相乘)种选法
  - 若`xxx = abc`
    1. `d < 1`，`abc1yyy > adc0efg`，超出n的范围，`0`种
    2. `d = 1`，`yyy = 000 ~ efg`，`efg + 1`种
    3. `d > 1`，`yyy = 000 ~ 999`，`1000`种

- 把上面全部的情况，累加起来，就是`1`出现在第四位的次数。

  类似的，可以求解出`1`**在任意一个位置上出现的次数，累加起来**，就求出了`1`在每一位上出现的此时，即求解出了`count(n,1)`。

- **注意：当`x=0`时**，不能有前导0，所以当`x=0`时，形如`xxx0yyy`，前面的`xxx`是从`001(不能从000开始，左边选法相比不是0的情况就少了一次)`到`999`

```c++
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

//求n的位数
int get(int  n){
    int res=0;
    while(n) res++,n/10;
    return res;
}

//求从1到n中i出现的次数
int count(int n,int i){
    int res=0,dgt=get(n);
    
    //遍历每一位 求出每一位出现i的次数
    for(int j=1;j<=dgt;j++){
        /*利用位运算
        得到当前遍历位次(第j位)的数的大小p：10^(j右边的位数即dgt-j)
        得到第j位左边的数大小l
        得到第j位右边的数大小r
        得到第j位上的数dj
        */
        int p=pow(10,dgt-j),l=n/p/10,r=n%p,dj=n/p%10;
        
        //然后分情况讨论 用i所在的位置划分出左边、右边
        /* 一、xxx..i..yyy的选法 左边取小于n中实际的左边的数l
        1)当i不为0时：左边000...~xxx...-1，右边...yyy就任意取了，取法：左边的数大小l*10^右边位数=l*p
        2)当i=0时，排除前导0的情况：左边000..1~xxx...-1，右边和上面一样，取法:(l-1)*p
        */
        if(i) res+=l*p;
        else res+=(l-1)*p;
        
        /* 二、左边固定为n的左边的数l
         1)、i > dj时 0种选法
         2)、i == dj时 yyy : 0...0 ~ r 即 r + 1 种选法
         3)、i < dj时 yyy : 0...0 ~ 9...9 即 10^(右边的数的位数) == p 种选法
        */
        if(i==dj) res+=r+1;
        if(i<dj) res+=p;
    }
    return res;
    
}

int main(){
    int a,b;
    while(cin>>a>>b,a){
        if(a>b) swap(a,b);//保证a为较小值，b为较大值
        for(int i=0;i<=9;i++)
            cout<<count(b,i)-count(a-1,i)<<' ';
        cout<<endl;
    }
    return 0;
}

```



## 6.状态压缩DP

**将一个十进制的整数转化为二进制数，每一位表示一种状态**

#### AcWing 291. 蒙德里安的梦想

![img](assets\356f9e87dbd432c34867befd3a9cd768.png)

**实现思路**：**先放横着的，再放竖着的。**

总方案数，等于**只放横着**的小方块的**合法方案数**。（放完横着的方块之后，竖着的只能被动填充进去，不需要额外进行竖着的情况）

方案合法的条件是：当横着的方块放完后，竖着的小方块恰好能把剩余的地方全部填满。

那如何判断方案是否合法呢？即怎么看竖着的小方块是否能把剩余部分填满呢？因为是竖着放的，所以可以按列来看，每一列的内部，只要所有**连续的空余小方块**的个数为**偶数**，即可。

- 状态表示**：`f[i][j]`表示前`i-1`列已经摆好，且从第`i-1`列是否有伸出一格到第`i`列的情况（这个情况用`j`表示）的所有方案数。**

  - 对于`j`：**`j`是一个二进制数**，**位数与棋盘的行数相等**，比如N=5，M=5的棋盘，`j`为5位的二进制数$(XXXXX)_2$，但**在程序中还是用十进制数表示**。（所以要用**位运算**来判断某一位是`1`或`0`）

  - **什么叫`j`表示从第`i-1`列伸出一格到第`i`列的情况？**如下图，第`i`列的第`1，2，5`个格子，是从`i-1`列伸过来的。此时的状态`j`为 $(11001)_2$，即对于第`i`列的所有格子，第`1，2，5`个格子被伸出来占据了（`j`是个二进制数，若该列的某一行，有被前面的列伸出来一格，则用`1`表示，否则用`0`表示），那么这些个位置被侵占了，就不能再横放了。

    ![img](assets\64622e706e67)

  - 这样对`f[i][j]`**更通俗的理解就是第`i`列的横放情况，由`j`的二进制表示出第`i`列哪些行可以横放（`j`的某个二进制位为0就表示当前行可以放）**

- **状态转移**：既然第` i `列固定了，我们需要看 第`i-2 `列是怎么转移到到第` i-1`列的（看最后转移过来的状态）。假设此时对应的状态是`k`（第`i-2`列到第`i-1`列伸出来的二进制数，比如00100），`k`也是一个二进制数，1表示哪几行小方块是横着伸出来的，0表示哪几行不是横着伸出来的。

  它对应的方案数是` f[i−1,k]`，即前`i-2`列都已摆完，且从第`i-2`列伸到第`i-1`列的状态为` k` 的所有方案数。

- 那`k`需要满足什么条件，才能够从`f[i - 1][k]`转移到`f[i][j]`呢？

  - ①**`k`和`j`不能冲突，也就是第`i-1`列和第`i`列的不能有重叠的1，即两列的相同行不能同时为1**（不能同时有前一列伸过来的）。如下图，在第一行`k`的二进制位为1，`j`的二进制位也为1，那么第`i`列第一行就不能再横放东西了。转化为代码判断就是**` (k & j ) ==0`** ，表示两个数相与，如果两有对应位同时为1，则相与结果为1，否则为0即没有冲突。

    ![image-20240804113657312](assets\image-20240804113657312.png)

  - ②既然从第`i-1`列到第`i`列横着摆的，和第`i-2`列到第`i-1`列横着摆的都确定了，那么第`i-1`列 空着的格子就确定了，这些空着的格子将来用作竖着放。**如果 某一列有这些空着的位置，那么该列所有连续的空着的位置长度必须是偶数（即不能有奇数个0）**。设置一个`st[]`数组表示的是这一列连续0的个数的情况，若为`true`表示连续偶数个0（合法状态），否则为`false`。体现到代码判断第`i-1`列满足竖放的合法状态：**`st[k|j]=true`**，

    - 解释：第`i-1`列中1的总个数，就是当前 第`i-1`列的到底有几个1，即哪几行是横着放格子的

      `st[k]`表明第`i-2`列插入到`i-1`列的1的个数，`i-1`列被动生成的1

      `st[j]`表明第`i-1`列插入到`i`列的1的个数，也就是`i-1`列自己生成的1

- 最后结果输出`f[m][0]`

注：对于`f[0][0]=1`和最后输出`f[m][0]`理解：数组下标从0开始表示实际棋盘的第一列，放`f[][]`表示当前列的横放情况,`f[0][0]`表示当前一列没有伸过来的（因为前面根本没有列），所以也就没有横放的情况，只有竖放的情况，即`f[0][0]=1`。所以输出`f[m][0]`表示第`m`列不横放，这也是合理的，如果第`m`列横放了，那就超出`m`列的范围了。

其他理解：[AcWing 291. 蒙德里安的梦想 - AcWing](https://www.acwing.com/file_system/file/content/whole/index/content/1634404/)

[AcWing 291. 蒙德里安的梦想 - AcWing](https://www.acwing.com/solution/content/239291/)

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N=12,M=1<<N;
long long f[N][M];
bool st[M];//记录当前连续0的个数是否为偶数，即合法状态true
int n,m;
int main(){
    
    while(cin>>n>>m,n||m){
        memset(f,0,sizeof f);//初始化为0
        //先预处理得到每一列连续0的个数 是偶数还是奇数
        for(int i=0;i<1<<n;i++){
            st[i]=true;
            int cnt=0;//记录0的个数
            
            for(int j=0;j<n;j++){//遍历当前列的每一行
                
                if(i>>j & 1){//取得当前行的一位 若为1
                    //判断之前记录的0的个数 是否为奇数
                    if(cnt & 1) st[i]=false;//若为奇数 标记为false 不合法
                }
                //取得当前行的一位 为0
                else cnt++;
            }
            //得到该列0个数的最终结果 判断是否是否合法
            if(cnt & 1) st[i]=false;//若个数为奇数 不合法
        }
        
        f[0][0]=1;
        
        //开始DP
        for(int i=1;i<=m;i++)//从列开始
            for(int j=0;j<1<<n;j++)//第i列情况
                for(int k=0;k<1<<n;k++)//第i-1列情况
                    if((j&k)==0 && st[j|k])//若没有冲突 且连续0的个数是偶数 
                        f[i][j]+=f[i-1][k];//合法
        
        cout<<f[m][0]<<endl;
    }
    return 0;
}

```



#### AcWing 91. 最短Hamilton路径

![img](assets\9a14b1b7cf85056051c187eb4b7147ce.png)

**实现思路**：![img](assets\e706e67252)

- `f(i,j)`表示从0号点到`j`号点，且中间经过点的状态是`i`（和上题类似，用二进制表示，位为1就表示某点经过，同样要进行**位运算**来获得每一位。如$ (1 1 1 0 1)_2 $ 表示第0，1，2，4被走过）的最短路径长度。
- 集合划分：最后一个点是`j`，按倒数第二个点划分，若倒数第二个点是`k`，即从0号点经过一系列不重复的点到达点`k`，再经过点`k`到达终点`j`。同时在这个状态`i`中就不能再包括点`j`了（因为路径要不重不漏），要再状态`i`中除去`j`。
- 最终状态转移方程为：**`f[i][j]=min(f[i][j],f[i-1<<j][k]+a[k][j])`；**因为`i`是二进制数，除去`j`点就表示使`j`位置上的二进制位由0后改为1，即相减。

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N=21,M=1<<21;
int n;
int f[M][M],a[N][N];
int main(){
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            cin>>a[i][j];
    
    memset(f,0x3f,sizeof f);//初始化图上各点距离为无穷大
    f[1][0]=0;//从0开始到0本身的距离为0
    
    for(int i=1;i<1<<n;i++)
        for(int j=0;j<n;j++)//枚举0到n号点
            if(i>>j & 1)//为1 要到达点j
                for(int k=0;k<n;k++)//枚举倒数第二个点k
                    if((i-(1<<j)) >> k)//如果k点在路径中
                        f[i][j]=min(f[i][j],f[i-(1<<j)][k]+a[k][j]);
    //从0 经过所有点 到达 n-1号点
    cout<<f[(1<<n)-1][n-1];
    
    return 0;
}
```



## 7.树形DP

#### AcWing 285. 没有上司的舞会

![img](assets\827d3918972b3c5ebee5c47986b3c0b6.png)

**实现思路**：所求即为从一棵中选出一个结点集合其各结点的值之和最大，且这个结点集合中都没有这些结点的父节点存在（没有父亲，只有同辈或者隔辈）

状态表示：利用递归

- `f(u, 0)`：从所有以`u`为根结点的子树中选择，并且**不选`u`这个点**的满足条件的最大快乐数

  不选择`u`这个点，那么看`u`的子节点`Si`，**子节点`Si`可以去，也可以不去**，那么再以这个子节点`Si`为根节点研究，看其去与不去情况下的快乐数选大者，**即`max(f[si][1],f[si][0])`，再将所有子节点的快乐数求和即为此状况下的最大快乐数**

- `f(u, 1)`：从所有以`u`为根结点的子树中选择，并且**选择`u`这个点**的满足条件的最大快乐数

  此时，`u`的子节点`Si`都不可以去，那么**只能看其所有子节点不去情况下`f[Si][0]`的快乐数，再求和为最终快乐数**

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N=6010;
int f[N][2],happy[N];//happy 各个结点的快乐数
int e[N],h[N],ne[N],idx;//使用邻接表存储树
bool has_father[N];//看i是否有父节点
int n;

void add(int a,int b){
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}

void dfs(int u){
    f[u][1]=happy[u];//初始为u自己的快乐数
    for(int i=h[u];i!=-1;i=ne[i]){
        int j=e[i];
        dfs(j);//对子节点继续递归
        f[u][0]+=max(f[j][0],f[j][1]);//父节点不去，得到所有子节点的两种情况的最大快乐数之和
        f[u][1]+=f[j][0];//父节点去，子结点不去的情况
    }
}


int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>happy[i];
    memset(h,-1,sizeof h);
    
    n-=1;//边的数量
    while(n--){
        int a,b;
        cin>>a>>b;
        has_father[a]=b;
        add(a,b);
    }
    
    int root=1;
    while(has_father[root]) root++;//遍历找到这课树的根节点
    
    dfs(root);
    //输出根节点去或不去两者的最大快乐数
    cout<<max(f[root][0],f[root][1]);
    return 0;
}

```



## 8.记忆化搜索

记忆化搜索就是对于某个子问题的解可能会被多次用到，那么就把子问题的解保存起来，以后用到的时候直接用，不用再计算一遍，以**滑雪**这道题为例，多条路径可能经过同一个点，那么就把这个点的值保存起来，只计算一次。

#### AcWing 901. 滑雪

![img](assets\d7352c5c6b6e35e96506765134e35be4.png)

![img](assets\5d5e37a79876129ff0350186a8a67307.png)

**实现思路**：

<img src="assets\20220329203216.png" alt="搜狗截图20220329203216.png" style="zoom:50%;" />

- 状态表示`f[i][j]`，表示从点`(i,j)`出发的最长路径长度
- 集合划分可分为四种情况：从点`(i,j)`出发，下一步的状态即为
  - 向左走到`(i,j-1)`，状态转换为`f[i][j-1]+1`
  - 向右走到`(i,j+1)`，状态转换为`f[i][j+1]+1`
  - 向上走到`(i-1,j)`，状态转换为`f[i-1][j]+1`
  - 向下走到`(i+1,j)`，状态转换为`f[i+1][j]+1`

- 注意：在向某个方向走时要判断是否走出边界以及所走的方向的值是否小于当前位置的值
- 没有给出固定出发点，所以要遍历每个点为起点出发

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N=310;
int f[N][N],h[N][N];
int n,m;
int dx={1,0,-1,0},dy={0,1,0,-1};//表示四个可走的方向

//以x,y为起点的最长路径长度 返回f[x][y]
int dp(int x,int y){
    if(f[x][y] !=-1) return f[x][y];//如果该点已经计算过了，就直接返回答案
    f[x][y]=1;//初始路径长度为1
    for(int i=0;i<4;i++){//四个方向选择
        int xx=x+dx[i],yy=y+dy[i];//选择一个方向走
        if(xx>=1 && xx<=n && yy>=1 && yy<=m && h[x][y]>h[xx][yy])
            f[x][y]=max(f[x][y],dp(xx,yy)+1);//继续递归更新
    }
    return f[x][y];//最后返回结果
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            cin>>h[i][j];
    memset(f,-1,sizeof f);//初始化f
    
    //以每个点为起点遍历
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            res=max(res,dp(x,y));
    cout<<res<<endl;
    return 0;
}
```



# 第六章 贪心算法

## 1.区间问题

#### **AcWing 905.区间选点**

![img](assets\cfcd7356b0e840dd9395ec1eb92a6998.png)

**实现思路**：

1. **将每个区间按照右端点从小到大排序**
2. 从前往后依次枚举每个区间
   - **若当前区间中已经包含点，则跳过**
   - **否则(即当前区间的左端点大于该点)，选择当前区间的右端点**

证明：比较最终结果`ans`和选出的点个数`cnt`大小关系，即证`ans>=cnt&&cnt>=ans`。

先证`ans<=cnt`：由于上述方法选择的方案保证了每一个区间都至少包含一个点，因此为一个合法的方案，而ans表示的是合法方案中的最少点个数，因此ans<=cnt。

再证`ans>=cnt`:考虑没有被跳过的区间，区间互不相交，因此选中cnt个区间，要想覆盖所有区间，最终的答案一定至少为cnt个点（因为区间是独立的），即ans>=cnt。得证。



```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=100010;

//使用一个结构体来表示区间
struct Range{
    int l,r;
    //重载一下比较符号 方便后续按照区间的右端点排序
    bool operator< (const Range &W)const{
        return r<W.r;
    }
}range[N];
int n;

int main(){
    cin>>n;
    for(int i=0;i<n;i++){
        int l,r;
        cin>>l>>r;
        range[i]={l,r};
    }
    
    sort(range,range+n);//使用sort按右端点对区间排序
    
    int res=0,st=-2e9;//st表示枚举的点
    for(int i=0;i<n;i++)
        if(range[i].l>st){//若区间的左端点大于当前点 即不包含该点
            st=range[i].r;//赋为右端点
            res++;
        }
    
    cout<<res;
    return 0;
    
}
```



#### **AcWing 908.最大不相交区间数量**

![img](assets\8b9626d94e69498c90ea074ab94e2680.png)

**实现思路：**与上题思路一致，代码也一样

 证明：比较最终结果ans和选出的区间个数cnt大小关系，即证ans>=cnt&&cnt>=ans。

先证ans>=cnt:由于选出的区间各不相交，因此为合法的方案，而ans为所有合法方案中最大的一个，因此有ans>=cnt。

再证ans<=cnt：运用反证法，假设ans>cnt，cnt表明每个区间都至少有一个选好的点，而ans表示所有不相交的区间的数量，说明至少应该有ans个点才能使每一个区间都有一个点，产生矛盾，因此ans<=cnt


```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=100010;
strucy Range{
    int l,r;
    bool operator<(const Range &W)const{
        return r<W.r;
    }
}range[N];

int n;
int main(){
    cin>>n;
    for(int i=0;i<n;i++){
        int l,r;
        cin>>l>>r;
        range[i]={l,r};
    }
    
    sort(range,range+n);
    
    int res=0,st=-2e9;
    for(int i=0;i<n;i++)
        if(range[i].l>st){
            st=range[i].r;
            res++;
        }
    cout<<res;
    return 0;
}

```



#### **AcWing 906.区间分组**

![img](assets\1758669a41d5442792e8cd7fa299f6dc.png)

**实现思路**：

1. 将所有区间按照**左端点从小到大排序**
2. 从前往后处理每个区间
   - 判断能否将当前区间放到某个现有的组当中：**判断现有组中的最后一个区间的右端点(即最大右端点)，是否大于当前区间的左端点**，若大于，则意味着该组存在一个区间与当前区间相交，则不能放到该组，需要重新开一个组，否则可以加入当前组。
3. **使用一个小根堆来存储所有组的右端点，那么堆顶就是右端点最小的一个组**，如果**当前区间的左端点小于这个组的右端点**，就表示当前区间会与现有组产生相交，**必然要新开一个组即加入堆中**。否则当前区间至少可以加入堆顶的那个组，更新一下右端点（就是删除堆顶元素，再添加）。
4. 最后输出堆的大小即为最小组数



```c++
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
using namespace std;
const int N=100010;

//使用一个结构体来表示区间
struct Range{
    int l,r;
    //重载一下 后续按照区间的左端点排序
    bool operator< (const Range &W)const{
        return l<W.l;
    }
}range[N];
int n;

int main(){
    cin>>n;
    for(int i=0;i<n;i++){
        int l,r;
        cin>>l>>r;
        range[i]={l,r};
    }
    
    sort(range,range+n);//使用sort按左端点对区间排序
    
    //使用优先队列构造小根堆
    priority_queue<int,vector<int>,greater<int> > heap;
    for(int i=0;i<n;i++){
        if(heap.empty() || heap.top()>=range[i].l) heap.push(range[i].r);//新开一个组
        else{//否则可以加入堆顶的组 更新一下右端点
            heap.pop();
            heap.push(range[i].r);
        }
    }
    
    cout<<heap.size();
    return 0;
    
}
```



#### **AcWing 907.区间覆盖**

![img](assets\302789f9c3564e6d9c759e8c23846147.png)

**实现思路**：

设线段的左端点为`start`，右端点为`end`

1. 将所有区间按照**左端点从小到大排序**
2. 从前往后依次枚举每个区间，在所有能覆盖`start`的区间中，选择一个右端点最大的区间，随后，将`start`更新为选中区间的右端点。当`start >= end`，结束
3. 用双指针算法来找**左端点<start，且右端点最大的区间**，若找到的右端点依旧小于`start`，即无解；否则区间数量+1，且更新`start`

![img](assets\737642e706e67)

注意：**一轮过后`i=j-1`**，`j`是满足条件的区间，**为了避免一些不必要的`i`枚举**，所以`i`可以跳到满足条件的区间继续向后，但因为一轮后`i++`，所以先`-1`，下一轮就从`j`开始，这样**又不会缺少或跳过满足的区间**

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=100010;

//使用一个结构体来表示区间
struct Range{
    int l,r;
    //重载一下 后续按照区间的左端点排序
    bool operator< (const Range &W)const{
        return l<W.l;
    }
}range[N];

int n;

int main(){
    int st,ed;//线段的起点和终点
    cin>>st>>ed;
    cin>>n;
    for(int i=0;i<n;i++)
    {
        int l,r;
        cin>>l>>r;
        range[i]={l,r};
    }
    sort(range,range+n);
    int res=0;
    
    bool success=false;//设置一个bool表示是否有解
    
    for(int i=0;i<n;i++){//先枚举每个区间
        int j=i,r=-2e9;//j找左端点小于等于start且右端点最大的区间 r表示这个最大右端点
        
        while(j<n && range[j].l<=st){
            r=max(r,range[j].r);
            j++;//后移继续找
        }
        
        //找到满足条件的区间
        if(r<st) break;//如果最大右端点还是小于线段起点 则无解
        //有解
        res++;// 区间数+1
        
        if(r>=ed){//找到的当前区间已经完全包含线段了 直接退出
            success=true;
            break;
        }
        
        //继续找下一个满足区间
        st=r;//更新起点
        
        i=j-1;//因为i一轮后++ 所以要先当前区间-1 下一次就从区间j开始 
    }
    if(success) cout<<res;
    else cout<<-1;
    return 0;
}
```



## 2.Huffman树-哈夫曼树

![img](assets\6b08b2e11e184f22b76e0bb7f8119189.png)

**实现思路**：构建一颗哈夫曼树，求最短带权路径长度（树中所有的叶结点的权值乘上其到根结点的路径长度）

- 每次选择重量最小的两堆进行合并
- 使用小根堆存储每一堆果子，每次两次弹出堆顶元素，合并后就放入小根堆中

```c++
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int main(){
    int n;
    cin>>n;
    priority_queue<int ,vector<int>,greater<int>> heap;
    while(n--){
        int x;
        cin>>x;
        heap.push(x);
    }
    
    int res=0;
    while(heap.size()>1){
        int a=heap.top(),heap.pop();
        int b=heap.top(),heap.pop();
        res+=a+b;
        heap.push(a+b);
    }
    cout<<res<<endl;
    return 0;
}
```



## 3.排序不等式

#### **AcWing 913.排队打水**

![img](assets\14625649b2ac439a8584e97d437effbe.png)

**实现思路**：

假设各个同学的打水时间为：`3 6 1 4 2 5 7` 并且就按照这个顺序来打水。 当第一个同学打的时候，后面所有同学都要等他，所以等待的总时长要加上一个`3 * 6`，第二个同学打的时候，后面所有同学也都要等他，所以要加上个`6 * 5`，以此类推，所有同学等待的总时长为`3 * 6 + 6 * 5 + 1 * 4 + 4 * 3 + 2 * 2 + 5 * 1`

假设各个同学打水花费的时长为 `t1，t2，t3，…，tn`，则按照次序打水，总的等待时长为：`t1 * (n-1) + t2 * (n-2) + ... + tn * 1`。

可以看出，**当打水顺序按照花费时间从小到大排序时，所得的等待时间最小**

> **证明**
>
> 采用反证法（调整法），假设最优解不是按照从小到大的顺序，则必然存在2个相邻的人，前一个人打水时长比后一个大，即必然存在一个位置i，满足t_i > t_i+1，那我们尝试把这两个同学的位置交换，看看对总的等待时长有什么影响，这两个同学的交换，只会影响他们两的等待时长，不会影响其他同学的等待时长。 交换前，这部分等待时长为`t_i * (n-i) + t_i+1 * (n-i-1)`，交换后，这部分等待时长为`t_i+1 * (n-i) + t_i * (n-i-1)`，容易算得，交换后的等待时长变小了，则该方案不是最优解，矛盾了。则最优解就是按照从小到大的顺序依次打水。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long LL;
const int N=100010;
int n,t[N];

int main(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>t[i];
    LL res=0;//数据范围比较大，防止溢出，使用LL
    sort(t,t+n);
    for(int i=0;i<n;i++) res+=t[i]*(n-i-1);
    cout<<res;
    return 0;
}
```



## 4.绝对值不等式

#### **AcWing 104.货仓选址**

![img](assets\e49970089f0c46aeac33e91b03449249.png)

**实现思路**：

假设`n`个商店在数轴上的坐标依次为：`x1`，`x2`，`x3`，…，`xn`

设仓库的位置为`x`，则总的距离为

```
f(x) = |x1 - x| + |x2 - x| + ... + |xn - x|
```

我们要求解的就是`f(x)`的最小值。

我们可以先进行一下分组，`1`和`n`为一组，`2`和`n-1`为一组…

```
f(x) = (|x1 - x| + |xn - x|) + (|x2 - x| + |x_n-1 - x|) + ....
```

单独看一组，任意一组都可以写成形如`|a - x| + |b - x|`的形式，`a`和`b`是已知的常数，`x`是未知数。假设`a < b`，则容易知道，当`x`取值在`[a,b]`这个区间内时，上面的表达式取得最小值`b - a`，而`x`取值只要落在`[a,b]`区间外，则上面的表达式的值一定是大于`b - a`的。

由此可知，对于分组`1`和`n`，只要`x`取值在`[x1,xn]`这个区间内，就能使`|x1 - x| + |xn - x|`取得最小值`xn - x1`。同理，对于`|x2 - x| + |x_n-1 - x|`，只要`x`取值在`[x2,x_n-1]`区间内，就能使这个部分取得最小值`x_n-1 - x2`…容易得出，**只要取所有分组的区间的交集即整个区间的中间点，能使总的`f(x)`最小。即，当`n`为偶数时，`x`只要落在最中间2个点之间即可；当`n`为奇数时，`x`只需要落在最中间的那个点上即可。**

```c++
#include <iostream>
#include <algorithm>
using namespace std;
const int N=100010;
int n,a[N];
int main(){
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
    sort(a,a+n);
    int res=0;
    for(int i=0;i<n;i++) res+=abs(a[i]-a[n/2]);
    cout<<res;
    return 0;
}
```



## 5.推公式

#### **AcWing** 125. 耍杂技的牛

![img](assets\5750cf59e30c45028367a731da170f90.png)

**实现思路**：wi表示牛的体重，si表示牛的强壮度

先给结论：**按照`w + s`从小到大的顺序，从上往下排，最大的危险系数一定是最小的。**

简单理解：把重量轻的牛放下面是很亏的，同样把不强壮的牛放下面也是亏的，所以就尽可能把又重又强壮的牛放下面。

**证明**：

从两方面证明：

- 按照上面策略得到的答案 `>=` 最优解
- 按照上面策略得到的答案 `<=` 最优解

首先，按照上面的策略得到的是一种方案，而最优解是所有方案中的最小值，所以 按照上面策略得到的答案 `>=` 最优解。

第二点，用反证法。假设最优解不是按照`w + s`从小到大排列。则一定存在一个位置`i`，使得`wi + si > w(i+1) + s(i+1)`，然后看一下把这两头牛交换，会发生什么变化。同样的，这两头牛的交换，不会影响除这两头牛以外，其他牛的危险系数，所以只看这两头牛的危险系数的变化即可。交换前和交换后，第`i`和第`i+1`头牛的危险系数如下

|        | 第i个位置上的牛               | 第i+1个位置上的牛                  |
| ------ | ----------------------------- | ---------------------------------- |
| 交换前 | w1 + w2 + … + w(i-1) - si     | w1 + w2 + … + wi - s(i+1)          |
| 交换后 | w1 + w2 + … + w(i-1) - s(i+1) | w1 + w2 + … + w(i-1) + w(i+1) - si |

去掉公共部分 `w1 + w2 + … + w(i-1)`

|        | 第i个位置上的牛 | 第i+1个位置上的牛 |
| ------ | --------------- | ----------------- |
| 交换前 | - si            | wi - s(i+1)       |
| 交换后 | - s(i+1)        | w(i+1) - si       |

随后，对所有项加上一个 `si + s(i+1)`，转化为正数，方便我们比较

|        | 第i个位置上的牛 | 第i+1个位置上的牛 |
| ------ | --------------- | ----------------- |
| 交换前 | s(i+1)          | wi + si           |
| 交换后 | s(i)            | w(i+1) + s(i+1)   |

由于`wi + si > si`，且根据先前的假设，有`wi + si > w(i+1) + s(i+1)`

所以`wi + si` 大于 `max(si, w(i+1) + s(i+1) )` ，进而有 `max( s(i+1), wi + si) > max(si, w(i+1) + s(i+1) )`。即，交换后，第`i`个和第`i+1`个位置上的牛中的最大危险系数变小了。

所以，只要存在一个位置`i`，使得`wi + si > w(i+1) + s(i+1)`，则一定能交换第`i`和第`i+1`的位置，使得总体的最大的危险系数不变或者变小。假设最优解不是按照`w + s`从小到大的顺序排列，则我们通过贪心策略，总是能够交换2个逆序的位置，使得到的结果，不变或者变得更小。因此，贪心得到的答案一定是 <= 最优解的。

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef pair<int,int> PII;//存储牛的体重+强壮度，体重
const int N=50010;
PII cow[N];
int n;
int main(){
    cin>>n;
    for(int i=0;i<n;i++){
        int w,s;
        cin>>w>>s;
        cow[i]={w+s,w};
    }
    
    sort(cow,cow+n);//自动先按体重+强壮度排序，再按体重排序
    int sum=0,res=-2e9;
    for(int i=0;i<n;i++){
        int w=cow[i].second,s=cow[i].first-w;//得到体重和强壮度
        res=max(res,sum-s);
        sum+=w;
    }
    cout<<res;
    return 0;
}
```

