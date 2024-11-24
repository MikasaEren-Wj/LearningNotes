## 哈希

### 1.两数之和

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例 2：**

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```
输入：nums = [3,3], target = 6
输出：[0,1]
```

**提示：**

- `2 <= nums.length <= 104`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`
- **只会存在一个有效答案**

**思路：**哈希表

最容易想到的方法是枚举数组中的每一个数` x`，寻找数组中是否存在 `target - x`。

当我们使用遍历整个数组的方式寻找 `target - x` 时，需要注意到每一个位于 `x` 之前的元素都已经和 `x` 匹配过，因此不需要再进行匹配。而每一个元素不能被使用两次，所以我们只需要在 `x` 后面的元素中寻找 `target - x`。

注意到上述方法时间复杂度较高的原因是寻找 `target - x` 的时间复杂度过高。因此，我们需要一种更优秀的方法，能够快速寻找数组中是否存在目标元素。如果存在，我们需要找出它的索引。

使用哈希表，可以将寻找` target - x `的时间复杂度降低到从 O(N) 降低到 O(1)。

这样我们创建一个哈希表，对于每一个` x`，我们首先查询哈希表中是否存在` target - x`，然后将 x 插入到哈希表中，即可保证不会让 `x`和自己匹配。

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> hashmap;
        for (int i = 0; i < nums.size(); i++){
            auto it=hashmap.find(target-nums[i]);
            if(it!=hashmap.end()){//如果存在会返回一个指向这个元素的迭代器;如果不存在，则返回hashmap.end()
                return {it->second,i};//不能使用'.',因为it是一个迭代器
            }
            hashmap[nums[i]]=i;
        }
        return {};
    }
};
```

~~~java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> hashTable=new HashMap<>();
        for(int i=0;i<nums.length;i++){
            if(hashTable.containsKey(target-nums[i]))//存在这个另一个数
                return new int[]{hashTable.get(target-nums[i]),i};//满足则直接返回
                //否则不存在这个匹配的数 先将这个数放入map中 继续遍历
                hashTable.put(nums[i],i);
        }
        return new int[0];
    }
}
~~~



### 49.字母异位词分组

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

 

**示例 1:**

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**示例 2:**

```
输入: strs = [""]
输出: [[""]]
```

**示例 3:**

```
输入: strs = ["a"]
输出: [["a"]]
```

 

**提示：**

- `1 <= strs.length <= 104`
- `0 <= strs[i].length <= 100`
- `strs[i]` 仅包含小写字母

方法一：哈希表+字符排序法

由于互为字母异位词的两个字符串包含的字母相同，因此对两个字符串分别进行排序之后得到的字符串一定是相同的，故可以将排序之后的字符串作为哈希表的键。

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>> mp;//哈希表，键为字符串排序后，值为互为字母异位词的集合
        for(string& str:strs){//加&引用实现高效访问
            string key=str;
            sort(key.begin(),key.end());//字符排序
            mp[key].emplace_back(str);//排序后拥有同样字符顺序的字符加入vector<string>中
        }
        vector<vector<string>> res;
        for(auto it=mp.begin();it!=mp.end();it++)//遍历哈希表
            res.emplace_back(it->second);
        return res;
    }
};
```

~~~java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        // 创建一个哈希表，键为排序后的字符串，值为对应的字母异位词列表
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] charArray = str.toCharArray();
            // 对字符串中的字符进行排序
            java.util.Arrays.sort(charArray);
            String sortedStr = new String(charArray);
            // 如果哈希表中已经存在该排序后的字符串作为键，将当前字符串添加到对应的值列表中
            if (map.containsKey(sortedStr)) {
                map.get(sortedStr).add(str);
            } else {
                // 如果不存在，创建新的列表，将当前字符串添加进去，并将键值对放入哈希表
                List<String> list = new ArrayList<>();
                list.add(str);
                map.put(sortedStr, list);
            }
        }
        // 返回哈希表中所有的值列表，即字母异位词分组后的结果
        return new ArrayList<>(map.values());
    }
}
~~~



方法二：哈希表+字母计数法

由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中的相同字母出现的次数一定是相同的，故可以将每个字母出现的次数使用字符串表示，作为哈希表的键。

由于字符串只包含小写字母，因此对于每个字符串，可以使用长度为 26 的数组记录每个字母出现的次数。需要注意的是，在使用数组作为哈希表的键时，不同语言的支持程度不同，因此不同语言的实现方式也不同。

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        /*
        哈希表
        键为字符串各个字母及其出现频率，如字符串bee
        那么键就是b1e2，以此判断是否为字母异位词 值为互为字母异位词的集合
        */
        unordered_map<string, vector<string>> mp;
        for (string& str : strs) { // 遍历每个字符串
            int count[26] = {0};   // 计数数组，记录每个字母的频率
            for (char c : str) {
                count[c - 'a']++;
            }
            string key = "";               // 键
            for (int i = 0; i < 26; i++) { // 遍历计数数组
                if (count[i]) {
                    key.push_back(i + 'a');  // 还原字母
                    key.push_back(count[i]); // 同时记录字母的频率
                }
            }
            mp[key].emplace_back(str); // 添加到哈希表
        }
        vector<vector<string>> res;
        for (auto& p : mp)
            res.emplace_back(p.second);
        return res;
    }
};
```

~~~java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map=new HashMap<>();
        for(String str:strs){
            int[] counts=new int[26];//统计每个字符串的各个字母数量
            for(int i=0;i<str.length();i++){
                counts[str.charAt(i)-'a']++;//对应位置计数
            }
            // 将每个出现次数大于0的字母和出现次数按顺序拼接成字符串，作为哈希表的键
            StringBuffer sb=new StringBuffer();
            for(int i=0;i<26;i++){
                if(counts[i]>0){
                    sb.append((char)('a'+i));
                    sb.append(counts[i]);
                }
            }
            String key=sb.toString();
            List<String> list=map.getOrDefault(key,new ArrayList<String>());//存在这个键就返回对应值，否则返回默认值即new ArrayList<String>()
            list.add(str);
            map.put(key,list);//放入map集合
        }
        return new ArrayList<List<String>>(map.values());//返回map的值集合
    }
}
~~~



### 128.最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

**示例 2：**

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

 

**提示：**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`

**思路：**

我们考虑枚举数组中的每个数 `x`，考虑以其为起点，不断尝试匹配 `x+1,x+2,⋯ `是否存在，假设最长匹配到了 `x+y`，那么以 x 为起点的最长连续序列即为 `x,x+1,x+2,⋯,x+y，`其长度为` y+1`，我们不断枚举并更新答案即可。

对于匹配的过程，暴力的方法是 O(n) 遍历数组去看是否存在这个数，但其实更高效的方法是用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化至 O(1) 的时间复杂度。

仅仅是这样我们的算法时间复杂度最坏情况下还是会达到 O(n^2)（即外层需要枚举 O(n) 个数，内层需要暴力匹配 O(n) 次），无法满足题目的要求。

但仔细分析这个过程，我们会发现其中执行了很多不必要的枚举，如果已知有一个` x,x+1,x+2,⋯,x+y `的连续序列，而我们却重新从 `x+1，x+2 `或者是 `x+y `处开始尝试匹配，那么得到的结果肯定不会优于枚举` x` 为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。

那么怎么判断是否跳过呢？由于我们要枚举的数 x 一定是在数组中不存在前驱数 `x−1 `的，不然按照上面的分析我们会从` x−1 `开始尝试匹配，因此我们每次在哈希表中检查是否存在 `x−1 `即能判断是否需要跳过了。

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if(nums.empty()) return 0;
        unordered_set<int> num_set;//设置一个去重哈希表
        for(const int& num:nums)
            num_set.insert(num);//存取哈希表 去重
        int res=0;//最长序列长度
        for(const int& num:num_set){
            /*
            如果存在比num小1的数，意味着从num开始枚举的连续序列必然不是最优的
            只有num前面没有num-1，那就可以考虑从num去枚举判断是不是最长
            */
            if(!num_set.count(num-1)){//可以从num开始枚举
                int currentNum=num;
                int currentlength=1;

                while(num_set.count(currentNum+1)){//开始判断以num开始的序列长度
                    currentNum++;
                    currentlength++;
                }
                res=max(res,currentlength);
            }
                
        }
        return res;
    }
};
```

~~~java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set=new HashSet<>();//可去重
        for(int num:nums){
            num_set.add(num);//先加入集合中
        }
        int result=0;
        for(int num:num_set){//遍历set集合
            if(!num_set.contains(num-1)){//如果不存在比当前小的数 就从当前数开始判断
            int currentNum=num;
            int currentResult=1;//长度默认1
            while(num_set.contains(currentNum+1)){//如果下一个数存在
                currentNum++;//递增1
                currentResult++;//长度+1
                }
            result=Math.max(result,currentResult);
            }
        }
        return result;
   }
}
~~~



## 双指针

### 283.移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

 

**示例 1:**

```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

**示例 2:**

```
输入: nums = [0]
输出: [0]
```

 

**提示**:

- `1 <= nums.length <= 104`
- `-231 <= nums[i] <= 231 - 1`

方法一：双指针找0和非0 互换

使用双指针，左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。

右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。

注意到以下性质：

1. 左指针左边均为非零数；

2. 右指针左边直到左指针处均为零。


因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
          if(nums.size()==0) return ;
          int i=0,j=0;//左指针指0，右指针指非0

          while(j<nums.size()){
            if( nums[j]){//右指针非0
                swap(nums[i],nums[j]);//交换
                i++;//左指针后移
            } 
            j++;//无论交换与否 右指针后移
            //若数组没有0 则i与j始终相等
          }
          
    }
};
```

方法二：值覆盖，非0的元素覆盖到前面，后面的元素直接置0

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int index = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != 0) {
                nums[index] = nums[i];//将非0元素移到前面
                index++;
            }
        }
        for (int i = index; i < nums.size(); i++) {//后面覆盖为0
            nums[i] = 0;
        }
    }
};
```

~~~java
class Solution {
    public void moveZeroes(int[] nums) {//值覆盖
        int noIndex=0;//非0元素下标
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=0){
                nums[noIndex++]=nums[i];
            }
        }
        for(int i=noIndex;i<nums.length;i++){//开始末尾赋值0
            nums[i]=0;
        }
    }
}
~~~



### 11.盛水最多的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

 

**示例 1：**

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

 

**提示：**

- `n == height.length`
- `2 <= n <= 105`
- `0 <= height[i] <= 104`

**思路：双指针**

设两指针 `i` , `j `，指向的水槽板高度分别为 `h[i] `, `h[j]` ，此状态下水槽面积为 `S(i,j) `。由于可容纳水的高度由两板中的 短板 决定，因此可得如下 面积公式 ：`S(i,j)=min(h[i],h[j])×(j−i)`

在每个状态下，无论长板或短板向中间收窄一格，都会导致水槽 底边宽度 −1 变短：

- 若向内 移动短板 ，水槽的短板 `min(h[i],h[j])`可能变大，因此下个水槽的面积 可能增大 。
- 若向内 移动长板 ，水槽的短板` min(h[i],h[j])` 不变或变小，因此下个水槽的面积 一定变小 。

因此，初始化双指针分列水槽左右两端，循环每轮将短板向内移动一格，并更新面积最大值，直到两指针相遇时跳出；即可获得最大面积。

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i=0,j=height.size()-1;
        int res=0;
        int m;
        while(i<j){
            m=min(height[i],height[j]);
            res=max(res,m*(j-i));
            if(height[i]<height[j]) i++;
            else j--;
        }
        return res;
    }
};
```

~~~java
class Solution {
    public int maxArea(int[] height) {
        int i=0,j=height.length-1;
        int result=0;
        while(i<j){
            int heigh=Math.min(height[i],height[j]);
            result=Math.max(result,heigh*(j-i));
            if(height[i]<height[j])
                i++;
            else
                j--;
        }
        return result;
    }
}
~~~



### 15.三数之和

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

 

**示例 1：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

**示例 2：**

```
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
```

**示例 3：**

```
输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
```

 

**提示：**

- `3 <= nums.length <= 3000`
- `-105 <= nums[i] <= 105`

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end()); // 对数组排序 防止出现重复组
        vector<vector<int>> res;
        for (int i = 0; i < n - 2;i++) { // 循环遍历，每次选择三元组的第一个元素，第一个元素只需枚举到n-3
            int x = nums[i]; // 记录第一个元素的值
            if (i > 0 && x == nums[i - 1])
                continue; // 每次枚举跳过上次已经枚举过的数
            if (x + nums[i + 1] + nums[i + 2] > 0)
                break; // 当前最小的三个元素已经大于0，没必要再枚举了，退出
            if (x + nums[n - 2] + nums[n - 1] < 0)
                continue; // 当前最小的元素+两个最大的元素小于0
                          // 那么当前元必然不满足 直接跳过哦本轮
            int j = i + 1,
                k = n - 1; // 左指针从前往后找第二个元，右指针从后往前找第三个元
            while (j < k) {
                int s = x + nums[j] + nums[k];
                if (s > 0)
                    --k; // 和大于0 右指针前移
                else if (s < 0)
                    j++; // 和小于0 左指针后移
                else {   // 和为0 满足
                    res.push_back({x, nums[j], nums[k]});
                    for (++j; j < k && nums[j] == nums[j - 1]; j++)
                        ; // 跳过重复元素
                    for (--k; k > j && nums[k] == nums[k + 1]; k--)
                        ; // 跳过重复元素
                }
            }
        }
        return res;
    }
};
```

~~~java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        int n=nums.length;
        Arrays.sort(nums);//先排序避免重复计算
        List<List<Integer>> ans=new ArrayList<>();
        for(int i=0;i<n-2;i++){//第一个数枚举到n-3即可
            int x=nums[i];
            if(i>0 && x==nums[i-1]) continue;//重复枚举 跳过
            if(x+nums[i+1]+nums[i+2]>0) break;//最小一组都大于0 直接退出
            if(x+nums[n-2]+nums[n-1]<0) continue;//最小的一个+最大两个小于0 直接下一轮
            //双指针开始找第二个 第三个元素
            int j=i+1,k=n-1;//左右指针
            while(j<k){
                int s=x+nums[j]+nums[k];
                if(s<0) j++;//左指针右移
                else if(s>0) k--;//右指针左移
                else{//和为0 找到一组
                    List<Integer> list=new ArrayList<>();
                    list.add(x);
                    list.add(nums[j]);
                    list.add(nums[k]);
                    ans.add(list);
                    for (++j; j < k && nums[j] == nums[j - 1]; j++)
                        ; // 跳过重复元素
                    for (--k; k > j && nums[k] == nums[k + 1]; k--)
                        ; // 跳过重复元素
                }
            }
        }
        return ans;
    }
}
~~~



## 滑动窗口

### 8.无重复字符的最长子串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长** **子串**的长度。



**示例 1:**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

 

**提示：**

- `0 <= s.length <= 5 * 104`
- `s` 由英文字母、数字、符号和空格组成

**思路：**滑动窗口

- 我们使用两个指针表示字符串中的某个子串（或窗口）的左右边界，其中左指针代表着「枚举子串的起始位置」，而右指针即为不重复子串（窗口）的终止位置；

- 在每一步的操作中，我们会将左指针向右移动一格，表示 我们开始枚举下一个字符作为起始位置，然后我们可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着 以左指针开始的，不包含重复字符的最长子串。我们记录下这个子串的长度；


在枚举结束后，我们找到的最长的子串的长度即为答案。

我们还需要使用一种数据结构来判断 是否有重复的字符，常用的数据结构为哈希集合（即 C++ 中的 `std::unordered_set`，Java 中的 `HashSet`，Python 中的 `set`, JavaScript 中的 `Set`）。在左指针向右移动的时候，我们从哈希集合中移除一个字符，在右指针向右移动的时候，我们往哈希集合中添加一个字符。

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

~~~java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set=new HashSet<>();//滑动窗口
        int n=s.length();
        int left=0,maxlen=0;//左指针和最大长度
        for(int i=0;i<n;i++){
            while(set.contains(s.charAt(i))){//循环删除窗口中存在的当前元素
                //窗口右移
                set.remove(s.charAt(left));//删除左指针的元素
                left++;//左指针右移一位
            }
            //不存在当前元素 加入窗口中
            maxlen=Math.max(maxlen,i-left+1);//更新最大窗口值
            set.add(s.charAt(i));//将当前元素加入窗口
        }
        return maxlen;
    }
}
~~~



### 9.找到字符串中所有字母的异位词 ###

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

 

**示例 1:**

```
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
```

 **示例 2:**

```
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
```

 

**提示:**

- `1 <= s.length, p.length <= 3 * 104`
- `s` 和 `p` 仅包含小写字母

**思路：**

根据题目要求，我们需要在字符串 s 寻找字符串 p 的异位词。因为字符串 p 的异位词的长度一定与字符串 p 的长度相同，所以我们可以在字符串 s 中构造一个长度为与字符串 p 的长度相同的滑动窗口，并在滑动中维护窗口中每种字母的数量；当窗口中每种字母的数量与字符串 p 中每种字母的数量相同时，则说明当前窗口为字符串 p 的异位词。

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int sl=s.size(),pl=p.size();
        vector<int> res;
        if(sl<pl) return res;
        //设置两个数组分别记录滑动窗口、字符串p中的字母出现的频率 大小为26
        vector<int> scount(26);
        vector<int> pcount(26);
        //初始化滑动窗口中的字母频率，记录p中字母频率 
        for(int i=0;i<pl;i++){
            scount[s[i]-'a']++;
            pcount[p[i]-'a']++;
        }
        if(scount==pcount) res.push_back(0);//s的首轮元素就和p满足异位
        for(int i=0;i<sl-pl;i++){//顶多遍历到倒数第pl个字母
            //窗口后移一位
            scount[s[i]-'a']--;//第一个移除
            scount[s[i+pl]-'a']++;//加入一个
            if(scount==pcount) res.push_back(i+1);//满足异位
        }
        return res;
    }
};
```

~~~java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int sl=s.length(),pl=p.length();
        List<Integer> result=new ArrayList<>();
        if(sl<pl) return result;
        //用两个数组分别记录滑动窗口、字符串p中的字母出现的频率 大小为26
        int[] slist=new int[26];
        int[] plist=new int[26];
        //数组初始化
        for(int i=0;i<pl;i++){
            slist[s.charAt(i)-'a']++;
            plist[p.charAt(i)-'a']++;
        }
        if(Arrays.equals(slist,plist)){//首轮元素就满足
            result.add(0);//添加起始索引
        }
        //继续比较
        for(int i=0;i<sl-pl;i++){//索引到sl-pl就行
            //移除第一个元素
            slist[s.charAt(i)-'a']--;
            //再添加一个元素
            slist[s.charAt(i+pl)-'a']++;
            //若两个窗口相等 添加起始索引
            if(Arrays.equals(slist,plist)) result.add(i+1);
        }   
        return result;
    }
}
~~~



## 子串

### 10.和为K的子数组 ###

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的子数组的个数* 。

子数组是数组中元素的连续非空序列。

 

**示例 1：**

```
输入：nums = [1,1,1], k = 2
输出：2
```

**示例 2：**

```
输入：nums = [1,2,3], k = 3
输出：2
```

 

**提示：**

- `1 <= nums.length <= 2 * 104`
- `-1000 <= nums[i] <= 1000`
- `-107 <= k <= 107`

**思路：**前缀和+哈希表

考虑以 `i` 结尾和为 `k` 的连续子数组个数，我们需要统计符合条件的下标 `j` 的个数,其中 `0≤j≤i` 且 `[j..i] `这个子数组的和恰好为` k `。我们知道对每个` i`，我们需要枚举所有的 `j` 来判断是否符合条件。考虑使用**前缀和和哈希表**进行优化。

我们定义 `pre[i]` 为 `[0..i] `里所有数的和，则 `pre[i]` 可以由 `pre[i−1] `递推而来，即：`pre[i]=pre[i−1]+nums[i]`
那么「`[j..i]` 这个子数组和为 `k` 」这个条件我们可以转化为根据**前缀和**，**某个区间和`pre[i]−pre[j−1]==k`**

简单移项可得符合条件的下标 `j` 需要满足**`pre[j−1]==pre[i]−k`**

所以我们考虑**以 `i` 结尾的和为 `k` 的连续子数组个数时只要统计有多少个前缀和为 `pre[i]−k` 的 `pre[j]` 即可**。我们建立哈希表 **mp，以和为键，出现次数为对应的值**，记录 `pre[i]` 出现的次数，从左往右边更新 mp 边计算答案，那么以 `i` 结尾的答案 `mp[pre[i]−k] `即可在 O(1) 时间内得到。最后的答案即为所有下标结尾的和为 `k` 的子数组个数之和。

需要注意的是，从左往右边更新边计算的时候已经保证了`mp[pre[i]−k]` 里记录的 `pre[j] `的下标范围是 `0≤j≤i `。同时，由于`pre[i]` 的计算只与前一项的答案有关，因此我们可以不用建立 `pre `数组，直接用 `pre` 变量来记录 `pre[i−1]` 的答案即可。

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        //哈希表 键为前面若干个数的和，值为这个和出现的次数
        unordered_map<int,int> mp;
        int res=0,pre=0;//pre用来计算前缀和
        mp[0]=1;//初始化和为0出现的次数为1
        for(auto& x:nums){//遍历数组 向后移动
            pre+=x;//计算前缀和
            /*
                如果前缀和中存在和为pre-k的情况（即某个点往前的和是pre-k），而当前的前缀和为pre，则存在某点到当前点的区间和为k，因为pre-(pre-k)=k
            */
            if(mp.find(pre-k)!=mp.end())//存在满足条件的前缀和
                res+=mp[pre-k];//加上这个和出现的次数
            mp[pre]++;//当前前缀和加入哈希表中
        }
        return res;
    }
};
```

~~~java
class Solution {
    public int subarraySum(int[] nums, int k) {
        //哈希表 键为前缀和 值为出现次数
        Map<Integer,Integer> mp=new HashMap<>();
        int res=0,pre=0;//结果和前缀和
        mp.put(0,1);//和为0 出现的次数为1
        for(int num:nums){
            pre+=num;//计算前缀和
            //查看前缀和中是否出现pre-k的情况
            if(mp.containsKey(pre-k)){
                res+=mp.get(pre-k);//存在就加上这个次数
            }
            //当前 前缀和 记录到哈希表中
            mp.put(pre,mp.getOrDefault(pre,0)+1);
        }
        return res;
    }
}
~~~



### 11.滑动窗口的最大值  ###

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

 

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```
输入：nums = [1], k = 1
输出：[1]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`
- `1 <= k <= nums.length`

**思路：**使用**单调队列**(C++双端队列)

- 设置一个滑动窗口队列(队头在左，队尾在右)，比较队尾和当前即将进入滑动窗口的数字大小，如果队尾数字小，则删除队尾。因此该队列中的数字全都大于待插入的数字，并且队头是最大的数字，再将该元素插入到队尾，最后输出队头即可。
- 使用C++STL自带的双端队列`deque`，**注意该数组存储的是对应元素的下标而不是实际元素值**。
- **每次都需先判断维持窗口队列的长度为k**，通过队头元素出队来维持，即寻找下一个窗口，然后得到最大值
- 循环比较待插入元素和队尾的关系，**两种情况**
  - ①队列不为空且待插元素一直比队尾元素大，则队尾元素不断出队。若直到队列为空，退出当前循环，让待插入元素存入队尾（此时队列仅一个元素）
  - ②若出现待插入元素比队尾元素小，那直接退出循环，让待插入元素存到队尾
  - **最终就使得队列中的元素是递减的**，最后输出队头元素即可

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        //队列 存储的是元素下标
        deque<int> q;
        for(int i=0;i<nums.size();i++){
            //维持窗口长度为k 若i-k+1（表示当前窗口第一个元素下标）>大于队头下标 则弹出队头（即头指针后移）
            while(!q.empty() && i-k+1>q.front()) q.pop_front();
            //构造单调递减队列
            while(!q.empty() && nums[q.back()]<=nums[i]) q.pop_back();//末尾元素小于当前元素 则不断移除末尾元素
            //直到找到合适的位置
            q.push_back(i);//存入下标
            //只要满足窗口长度达到k
            if(i>=k-1) res.push_back(nums[q.front()]);//队头元素 即为当前窗口最大值
        }
        return res;
    }
};
```

~~~java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n=nums.length;
        Deque<Integer> deque=new LinkedList<>();//双端队列 存储的是元素的下标
        int[] res=new int[n-k+1];
        for(int i=0;i<n;i++){
            //维持窗口的长度为k 超过k长度则队头元素出队
            while(!deque.isEmpty() && i-k+1>deque.peekFirst()) 
                deque.pollFirst();//移除队头元素
            //构造单调递减队列
            while(!deque.isEmpty() && nums[i]>nums[deque.peekLast()])
                //如果当前元素大于队尾元素 则移除队尾元素
                deque.pollLast();
            //找到合适位置 向队尾加入该元素
            deque.offerLast(i);
            //再次查看窗口是否满足长度为k 若满足则添加队头元素到结果数组
            if(i-k+1>=0) 
                res[i-k+1]=nums[deque.peekFirst()];
        }
        return res;
    }
}
~~~



### 12.最小覆盖子串 ###

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。



**注意：**

- 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量。
- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

 

**示例 1：**

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
```

**示例 2：**

```
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
```

**示例 3:**

```
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
```

 

**提示：**

- `m == s.length`
- `n == t.length`
- `1 <= m, n <= 105`
- `s` 和 `t` 由英文字母组成

**思路**：滑动窗口+双指针

本问题要求我们返回字符串` s `中包含字符串 `t `的全部字符的最小窗口。我们称包含 `t` 的全部字母的窗口为**「可行」窗口。**

我们可以用滑动窗口的思想解决这个问题。在滑动窗口类型的问题中都会有两个指针，一个用于「延伸」现有窗口的 `r `指针，和一个用于「收缩」窗口的 `l `指针。

**在任意时刻，只有一个指针运动，而另一个保持静止**。

我们在 `s` 上滑动窗口，通过移动 `r` 指针不断扩张窗口。当窗口包含` t` 全部所需的字符后，如果能收缩，我们就使用`l`指针收缩窗口直到得到最小窗口。

如何判断当前的窗口包含所有 `t` 所需的字符呢？

我们可以用一个哈希表表示 `t `中所有的字符以及它们的个数，用一个哈希表动态维护窗口中所有的字符以及它们的个数，如果这个动态表中包含 `t` 的哈希表中的所有字符，并且对应的个数都不小于 `t` 的哈希表中各个字符的个数，那么当前的窗口是「可行」的。

```c++
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char,int> tmp,smp;
        string res=s;//初始化结果串
        //左指针(用于收缩) 滑动窗口内有效字母的个数（有效就是存在于t中的）
        int left=0,correct=0;
        //对于字符t制作记录其字母的哈希表
        for(auto& item:t){
            tmp[item]++;//键为字母，值为对应出现次数
        }

        //右指针用于延伸
        for(int right=0;right<s.size();right++){
            //更新滑动窗口smp 记录加入的字母及其次数
            smp[s[right]]++;
            //添加过后 滑动窗口中对应字母的次数还小于t中对应字母的次数 添加是有效的 
            if(tmp[s[right]]>=smp[s[right]]) correct++;//更新记录有效

            //更新左指针 即收缩 去掉多余的字母
            while(left<right && tmp[s[left]]<smp[s[left]])
                smp[s[left++]]--;//次数多余 减去 记得左指针要右移
            
            if(correct==t.size()){//滑动窗口已经包含t
                if(right-left+1<res.size())//如果此时滑动窗口比之前更小
                    res=s.substr(left,right-left+1);//更新结果res
            }
            
        }
        //如果correct刚好为t的大小 意味着存在结果 否则不存在返回空字符串
        return correct==t.size()?res:"";
    }
};
```

## 普通数组

### 13.最大子数组的和 ###

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**是数组中的一个连续部分。

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**示例 2：**

```
输入：nums = [1]
输出：1
```

**示例 3：**

```
输入：nums = [5,4,-1,7,8]
输出：23
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`

**思路：**动态规划

假设 nums 数组的长度是 n，下标从 0 到 n−1。

我们用 `f(i)` 代表以第 `i` 个数结尾的「连续子数组的最大和」，那么很显然我们要求的答案就是：`max {f(i)} 0≤i≤n−1`

因此我们只需要求出每个位置的 `f(i)`，然后返回 f 数组中的最大值即可。

那么我们如何求 `f(i)` 呢？

我们可以考虑 `nums[i]` 单独成为一段还是加入 `f(i−1)` 对应的那一段，这取决于 `nums[i]` 和 `f(i−1)+nums[i]` 的大小，我们希望获得一个比较大的，于是可以写出这样的动态规划转移方程：`f(i)=max{f(i−1)+nums[i],nums[i]}`

不难给出一个时间复杂度 O(n)、空间复杂度 O(n) 的实现，即用一个 f 数组来保存 `f(i)` 的值，用一个循环求出所有` f(i)`。

考虑到 `f(i)` 只和 `f(i−1)` 相关，于是我们可以只用一个变量 pre 来维护对于当前 `f(i)` 的 `f(i−1)` 的值是多少，从而让空间复杂度降低到 O(1)，这有点类似「滚动数组」的思想。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int pre=0,res=nums[0];
        for(auto& x:nums){
            //f(i-1)+x 可能x加上f(i-1)反而降低了x自身的身价 
            pre=max(pre+x,x);//取大者 x是选择接收前面或者独立出来
            res=max(pre,res);//更新最大值
        }
        return res;
    }
};
```

方法二：前缀和

https://leetcode.cn/problems/maximum-subarray/solutions/2533977/qian-zhui-he-zuo-fa-ben-zhi-shi-mai-mai-abu71

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        //pre为前缀和，pre_min为前缀和中的最小值
       int res=-1e9,pre=0,pre_min=0;
       for(auto& x:nums){
            pre+=x;//计算当前前缀和
            res=max(res,pre-pre_min);//更新 减去前面最小的前缀和
            pre_min=min(pre,pre_min);//更新最小的前缀和
       }
       return res;
    }
};
```

### 14.合并区间

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

 

**示例 1：**

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

**示例 2：**

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

 

**提示：**

- `1 <= intervals.length <= 104`
- `intervals[i].length == 2`
- `0 <= starti <= endi <= 104`

**思路**：

![Image](D:/桌面/za/算法/assets/e706e67)

> - 对于第一种情况，区间不变
> - 对于第二种情况，end要变成区间i的右端点
>   - 前面两种情况，可以合并为将end更新为end和区间`i`的右端点中的较大者
> - 对于第三种情况，将当前维护的区间加入答案，并将维护的区间更新为区间`i`

```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        //对所有区间按左端点排序
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> res;
        //表示前一个区间左右端点指针 初始为-1
        int st=-1,ed=-1;
        for(auto& q:intervals){
            if(ed<q[0]){//若当前区间与前一个区间无交集
                //当前区间加入结果数组 注意初始遍历第一个区间时不立即加入
                if(st!=-1) res.push_back({st,ed});
                st=q[0],ed=q[1];//更新前一个区间的左右端点指针
            }else{
                //否则当前区间与前一个区间有交集 合并区间
                ed=max(ed,q[1]);//st不变 ed更新为前面和当前区间右端点更大值
            }
        }
        if(st!=-1) res.push_back({st,ed});
        return res;
    }
};
```

### 15.转轮数组 ###

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

 

**示例 1:**

```
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
```

**示例 2:**

```
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-231 <= nums[i] <= 231 - 1`
- `0 <= k <= 105`

**思路**：

方法一：额外的数组

```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
            int n=nums.size();
            vector<int> a(n);
            for(int i=0;i<n;i++){
                a[(i+k)%n]=nums[i];
            }
            nums.assign(a.begin(),a.end());
    }
};
```

方法二：数组反转

当我们将数组的元素向右移动 `k` 次后，尾部 `k mod n` 个元素会移动至数组头部，其余元素向后移动 `k mod n` 个位置。

该方法为数组的翻转：我们可以先将所有元素翻转，这样尾部的 `k mod n` 个元素就被移至数组头部，然后我们再翻转 `[0,k mod n−1]` 区间的元素和 `[k mod n,n−1]` 区间的元素即能得到最后的答案。

```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
            k%=nums.size();
            reverse(nums.begin(),nums.end());
            reverse(nums.begin(),nums.begin()+k);
            reverse(nums.begin()+k,nums.end());
    }
};
```

### 16.除自身以外的数组乘积

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请 **不要使用除法，**且在 `O(n)` 时间复杂度内完成此题。

 

**示例 1:**

```
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

**示例 2:**

```
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
```

 

**提示：**

- `2 <= nums.length <= 105`
- `-30 <= nums[i] <= 30`
- **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内

**思路**：

方法一：初始化两个空数组 `pre` 和 `ne`。对于给定索引 `i`，`pre[i]` 代表的是 `i` 左侧所有数字的乘积，`ne[i]` 代表的是 `i` 右侧所有数字的乘积。

```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n=nums.size();
        //前缀积（不包含自身）
        vector<int> pre(n);
        //后缀积（不包含自身）
        vector<int> ne(n);
        //结果数组
        vector<int> ans(n);
        pre[0]=ne[n-1]=1;//初始化0位置为1
        for(int i=1,j=n-2;i<n;i++,j--){
            pre[i]=pre[i-1]*nums[i-1];
            ne[j]=ne[j+1]*nums[j+1];
        }
        //得到结果 为前缀之积*后缀之积
        for(int i=0;i<n;i++){
            ans[i]=pre[i]*ne[i];
        }
        return ans;
    }
};
```

方法二：由于输出数组不算在空间复杂度内，那么我们可以将 pre或 ne 数组用输出数组来计算。先把输出数组当作 pre 数组来计算，然后再动态构造 ne 数组得到结果。

```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n=nums.size();
        vector<int> ans(n);
        ans[0]=1;//初始化0位置为1
        //先算索引i前缀积 直接先用结果数组ans[i]存储
        for(int i=1;i<n;i++){
            ans[i]=ans[i-1]*nums[i-1];
        }
        //r表示i后缀积 算后缀积的同时得到结果
        int r=1;//初始化为1
        for(int i=n-1;i>=0;i--){
            ans[i]=ans[i]*r;//对于i 前缀积ans[i]  后缀积r
            r*=nums[i];//同步更新后缀积
        }
        return ans;
    }
};
```

### 17.缺失的第一个正数

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

 

**示例 1：**

```
输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。
```

**示例 2：**

```
输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，但 2 没有。
```

**示例 3：**

```
输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正数 1 没有出现。
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-231 <= nums[i] <= 231 - 1`

**思路：**

先对数组进行排序，初始化未出现的最小正数`res=1`（最小正整数开始）。遍历数组，寻找数组中的未比较的最小正数`num[i]`，和res比较，三种情况：

- 大于`res`，则直接返回`res`即为最终结果
- 与`res`相等，则`res++`，因为此时最小正数已经出现在数组中了
- 小于`res`，`res`不处理，继续遍历，因为后续可能出现大于等于`res`的情况

遍历完数组，返回res即为最终结果

```c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int res=1;//未出现的最小正整数 从1开始枚举
        for(int i=0;i<nums.size();i++){
            //找数组中的最小正整数
            if(nums[i]<=0) continue;
            //然后和res比较
            if(nums[i]>res) return res;//大于res 直接返回res
            else if(nums[i]==res) res++;//等于res res++
            //小于res 不处理 因为后续可能出现大于等于res的情况或直至遍历完都小于res
        }
        return res;
    }
};
```

## 矩阵

### 18.矩阵置零 

给定一个 `*m* x *n*` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法**。**

 

**示例 1：**

<img src="assets\mat1.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
```

**示例 2：**

<img src="assets\mat2.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

**提示：**

- `m == matrix.length`
- `n == matrix[0].length`
- `1 <= m, n <= 200`
- `-231 <= matrix[i][j] <= 231 - 1`



**思路**：

**方法一：使用标记数组**

我们可以用两个标记数组分别记录每一行和每一列是否有零出现。

具体地，我们首先遍历该数组一次，如果某个元素为 0，那么就将该元素所在的行和列所对应标记数组的位置置为 true。最后我们再次遍历该数组，用标记数组更新原数组即可。

```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m=matrix.size();//行数
        int n=matrix[0].size();//列数
        vector<int> row(m),col(n);//标记某行或某列出现零
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                if(!matrix[i][j])
                    row[i]=col[j]=1;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                if(row[i] || col[j])
                    matrix[i][j]=0;

    }
};
```

**方法二：使用两个标记变量**

我们可以用矩阵的第一行和第一列代替方法一中的两个标记数组，以达到 O(1) 的额外空间。但这样会导致原数组的第一行和第一列被修改，无法记录它们是否原本包含 0。因此我们需要额外使用两个标记变量分别记录第一行和第一列是否原本包含 0。

在实际代码中，我们首先预处理出两个标记变量，接着使用其他行与列去处理第一行与第一列，然后反过来使用第一行与第一列去更新其他行与列，最后使用两个标记变量更新第一行与第一列即可。

```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();          // 行数
        int n = matrix[0].size();       // 列数
        int flag_col = 0, flag_row = 0; // 标记第一行或第一列是否出现0
        // 先预处理出第一行和第一列是否含有0
        for (int i = 0; i < m; i++)
            if (!matrix[i][0])
                flag_col = 1;
        for (int i = 0; i < n; i++)
            if (!matrix[0][i])
                flag_row = 1;
        // 从1开始遍历 用第一行第一列来标记
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                if (!matrix[i][j])
                    matrix[i][0] = matrix[0][j] = 0;
        // 从1开始 遍历置0
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                if (!matrix[i][0] || !matrix[0][j])
                    matrix[i][j] = 0;
        // 再处理第一行 第一列
        if (flag_col)
            for (int i = 0; i < m; i++)
                matrix[i][0] = 0;
        if (flag_row)
            for (int i = 0; i < n; i++)
                matrix[0][i] = 0;
    }
};
```

### 19.螺旋矩阵 ###

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

**示例 1：**

<img src="assets\spiral1.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

<img src="assets\spiral.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

**思路**：

这里的方法不需要记录已经走过的路径，所以执行用时和内存消耗都相对较小

1. 首先设定上下左右边界

2. 其次向右移动到最右，此时第一行因为已经使用过了，可以将其从图中删去，体现在代码中就是重新定义上边界
3. 判断若重新定义后，上下边界交错，表明螺旋矩阵遍历结束，跳出循环，返回答案
4. 若上下边界不交错，则遍历还未结束，接着向下向左向上移动，操作过程与第一，二步同理
5. 不断循环以上步骤，直到某两条边界交错，跳出循环，返回答案

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();         
        int n = matrix[0].size();
        vector<int> res;
        //设定上下左右四个边界
        int u=0,d=m-1;//上下
        int l=0,r=n-1;//左右

        while(true){
            //上边界 从左遍历到右
            for(int i=l;i<=r;i++) res.push_back(matrix[u][i]);
            //更新上边界 同时判断上边界是否大于下边界
            if(++u>d) break;//退出
            //右边界 从上遍历到下
            for(int i=u;i<=d;i++) res.push_back(matrix[i][r]);
            //更新右边界 判断
            if(--r<l) break;
            //下边界 从右遍历到左
            for(int i=r;i>=l;i--) res.push_back(matrix[d][i]);
            //更新下边界 判断
            if(--d<u) break;
            //左边界 从下遍历到上
            for(int i=d;i>=u;i--) res.push_back(matrix[i][l]);
            //更新左边界 判断
            if(++l>r) break;
        }
        return res;
    }
};
```

### 20.旋转图像

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**原地** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

 

**示例 1：**

<img src="assets\545613531.jpg" alt="img" style="zoom: 80%;" />

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**示例 2：**

<img src="assets\56466.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

 

**提示：**

- `n == matrix.length == matrix[i].length`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

**思路：**

1. 先将矩阵转置
2. 再将左右对称的两列互换

```c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        int temp=0;//存储临时值

        //矩阵转置
        for(int i=0;i<n;i++)
            for(int j=i+1;j<n;j++){
                temp=matrix[i][j];
                matrix[i][j]=matrix[j][i];
                matrix[j][i]=temp;
            }
        //对称列互换
        for(int i=0;i<n/2;i++)
            for(int j=0;j<n;j++){
                temp=matrix[j][i];
                matrix[j][i]=matrix[j][n-1-i];
                matrix[j][n-1-i]=temp;
            }
    }
};
```

### 21.搜索二维矩阵  ###

编写一个高效的算法来搜索 `*m* x *n*` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

 

**示例 1：**

<img src="assets\searchgrid2.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```

**示例 2：**

<img src="assets\searchgrid.jpg" alt="img" style="zoom:67%;" />

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= n, m <= 300`
- `-109 <= matrix[i][j] <= 109`
- 每行的所有元素从左到右升序排列
- 每列的所有元素从上到下升序排列
- `-109 <= target <= 109`

**思路**：

以 matrix 中的 **左下角元素**为标志数 **flag** ，则有:

若 flag > target ，则 target 一定在 flag 所在 行的上方 ，即 flag 所在行可被消去。
若 flag < target ，则 target 一定在 flag 所在 列的右方 ，即 flag 所在列可被消去。

流程：

1. 从矩阵 matrix 左下角元素（索引设为 (i, j) ）开始遍历，并与目标值对比：

   - 当 `matrix[i][j] `> `target `时，执行 i-- ，即消去第 i 行元素。

   - 当` matrix[i][j] `< `target` 时，执行 j++ ，即消去第 j 列元素。

   - 当 `matrix[i][j] `=` target` 时，返回 true ，代表找到目标值。

2. 若行索引或列索引越界，则代表矩阵中无目标值，返回 false 。

> 每轮 i 或 j 移动后，相当于生成了“消去一行（列）的新矩阵”， 索引(i,j) 指向新矩阵的左下角元素（标志数），因此可重复使用以上性质消去行（列）。

> [240. 搜索二维矩阵 II - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix-ii/solutions/2361487/240-sou-suo-er-wei-ju-zhen-iitan-xin-qin-7mtf/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int i=matrix.size()-1,j=0;//左下角元素开始
        while(i>=0 && j<matrix[0].size()){
            if(matrix[i][j]>target) i--;//消去一行
            else if(matrix[i][j]<target) j++;//消去一列
            else return true;
        }
        return false;
    }
};
```

## 链表

### 160.相交链表  ###

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

图示两个链表在节点 `c1` 开始相交**：**

[![img](assets\160_statement.png)

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

**自定义评测：**

**评测系统** 的输入如下（你设计的程序 **不适用** 此输入）：

- `intersectVal` - 相交的起始节点的值。如果不存在相交节点，这一值为 `0`
- `listA` - 第一个链表
- `listB` - 第二个链表
- `skipA` - 在 `listA` 中（从头节点开始）跳到交叉节点的节点数
- `skipB` - 在 `listB` 中（从头节点开始）跳到交叉节点的节点数

评测系统将根据这些输入创建链式数据结构，并将两个头节点 `headA` 和 `headB` 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 **视作正确答案** 。

 

**示例 1：**

<img src="assets\160_example_1_1.png" alt="img" style="zoom: 80%;" />

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
— 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。
```

 

**示例 2：**

<img src="assets\160_example_2.png" alt="img" style="zoom:80%;" />

```
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
```

**示例 3：**

<img src="assets\160_example_3.png" alt="img" style="zoom:80%;" />

```
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。
```

**提示：**

- `listA` 中节点数目为 `m`
- `listB` 中节点数目为 `n`
- `1 <= m, n <= 3 * 104`
- `1 <= Node.val <= 105`
- `0 <= skipA <= m`
- `0 <= skipB <= n`
- 如果 `listA` 和 `listB` 没有交点，`intersectVal` 为 `0`
- 如果 `listA` 和 `listB` 有交点，`intersectVal == listA[skipA] == listB[skipB]`

**思路**：

根据题目意思：**如果两个链表相交，那么相交点之后的长度是相同的**

要做的事情是：让两个链表的指针**从距离末尾 同等距离 的位置开始遍历**。这时短链表的指针是指向其头节点的，而长链表的指针指向的结点及其后面结点的数量是和短链表的长度相同的

实现：

- 指针 pA 指向 A 链表，指针 pB 指向 B 链表，同时往后遍历
- 如果 pA 到了末尾，则令 pA = headB 继续遍历
- 如果 pB 到了末尾，则 令pB = headA 继续遍历

假设A链表比B链表长，那么pB先到达末尾，然后从A链表继续遍历。直到pA到达A链表的末尾，然后指向B链表的头结点。**此时pB和pB距离末尾的距离是相等的。从此时开始同时往后遍历，判断是否相交**

> 为什么会出现这种情况？
>
> 因为A 链表 + B 链表，与 B 链表 + A 链表必然是相同的长度，所以两个指针分别交替遍历一次两个链表后必然同时到达末尾。
>
> [160. 相交链表 - 力扣（LeetCode）](https://leetcode.cn/problems/intersection-of-two-linked-lists/solutions/10774/tu-jie-xiang-jiao-lian-biao-by-user7208t/?envType=study-plan-v2&envId=top-100-liked)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA==NULL || headB==NULL) return NULL;
        ListNode *pA=headA,*pB=headB;//初始化两指针
        while(pA!=pB){
            // 如果 pA 到达链表末尾，则将其指向链表 B 的头部
            pA= pA==NULL?headB:pA->next;
            // 如果 pB 到达链表末尾，则将其指向链表 A 的头部
            pB= pB==NULL?headA:pB->next;
        }
        //直至两者相等或同时到达末尾
        return pA; // 如果 pA 和 pB 相遇，则返回交点节点；否则，返回 null
    }
};
```

### 206.反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

 

**示例 1：**

![img](assets\rev1ex1.jpg)

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**示例 2：**

![img](assets\rev1ex2.jpg)

```
输入：head = [1,2]
输出：[2,1]
```

**示例 3：**

```
输入：head = []
输出：[]
```

 

**提示：**

- 链表中节点的数目范围是 `[0, 5000]`
- `-5000 <= Node.val <= 5000`

**思路**：

假设链表为 1→2→3→∅，我们想要把它改成 ∅←1←2←3。

在遍历链表时，将当前节点的 `next` 指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头引用。

故用三个指针，`pre`指向当前结点的前驱，`cur`指向当前结点，`next`指向当前结点的后继

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre=nullptr;
        ListNode* cur=head;
        while(cur){
            ListNode* next=cur->next;
            //当前结点各指针反转
            cur->next=pre;//后继指针指向前驱
            pre=cur;//前驱指针指向当前结点
            cur=next;//后移
        }
        return pre;//返回最后一个节点 也就是新链表的头结点
    }
};
```

### 234.回文链表

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。**回文** 序列是向前和向后读都相同的序列。

**示例 1：**

![img](assets\pal1linked-list.jpg)

```
输入：head = [1,2,2,1]
输出：true
```

**示例 2：**

![img](assets\pal2linked-list.jpg)

```
输入：head = [1,2]
输出：false
```

 

**提示：**

- 链表中节点数目在范围`[1, 105]` 内
- `0 <= Node.val <= 9`

**思路：**

方法一：

1. 复制链表值到数组列表中。
2. 使用双指针法判断是否为回文

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> arr;
        ListNode* cur=head;
        while(cur){
            arr.push_back(cur->val);
            cur=cur->next;
        }
        for(int i=0,j=arr.size()-1;i<j;i++,j--){
            if(arr[i]!=arr[j]) return false;
        }
        return true;
    }
};
```

方法二：

将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较。比较完成后我们应该将链表恢复原样。虽然不需要恢复也能通过测试用例，但是使用该函数的人通常不希望链表结构被更改。

整个流程可以分为以下五个步骤：

- 找到前半部分链表的尾节点。

- 反转后半部分链表。
- 判断是否回文。
- 恢复链表。
- 返回结果。

执行步骤一，我们可以计算链表节点的数量，然后遍历链表找到前半部分的尾节点。

我们也可以使用快慢指针在一次遍历中找到：慢指针一次走一步，快指针一次走两步，快慢指针同时出发。当快指针移动到链表的末尾时，慢指针恰好到链表的中间。通过慢指针将链表分为两部分。

若链表有奇数个节点，则中间的节点应该看作是前半部分。

步骤二可以使用「206. 反转链表」问题中的解决方法来反转链表的后半部分。

步骤三比较两个部分的值，当后半部分到达末尾则比较完成，可以忽略计数情况中的中间节点。

步骤四与步骤二使用的函数相同，再反转一次恢复链表本身。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //反转链表
    ListNode* reverseList(ListNode* head){
        ListNode* pre=nullptr;
        ListNode* cur=head;
        while(cur){
            ListNode* next=cur->next;
            cur->next=pre;
            pre=cur;
            cur=next;
        }
        return pre;
    }

    //找到前半部分的尾结点
    ListNode* EndOffirstHalf(ListNode* head){
        //使用快慢指针
        ListNode* fast=head;
        ListNode* slow=head;
        while(fast->next && fast->next->next){
            fast=fast->next->next;
            slow=slow->next;
        }
        //fast指针走到末尾 此时slow即指向前半部分的尾结点
        return slow;
    }

    bool isPalindrome(ListNode* head) {
        ListNode* firstHalfEnd=EndOffirstHalf(head);
        //后半部分反转
        ListNode* secondHalfFirst=reverseList(firstHalfEnd->next);

        //判断是否为回文
        ListNode* p1=head;
        ListNode* p2=secondHalfFirst;
        bool res=true;
        while(res && p2){
            if(p1->val!=p2->val) res=false;
            p1=p1->next;
            p2=p2->next;
        }
        //还原链表
        firstHalfEnd->next=reverseList(secondHalfFirst);
        return res;
    }
};
```

### 141.环形链表 ###

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

*如果链表中存在环* ，则返回 `true` 。 否则，返回 `false` 。

 

**示例 1：**

![img](assets\circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![img](assets\circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![img](assets\circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

 

**提示：**

- 链表中节点的数目范围是 `[0, 104]`
- `-105 <= Node.val <= 105`
- `pos` 为 `-1` 或者链表中的一个 **有效索引** 。



**思路**：

方法一：遍历所有节点，每次遍历到一个节点时，判断该节点此前是否被访问过。

具体地，我们可以使用哈希表来存储所有已经访问过的节点。每次我们到达一个节点，如果该节点已经存在于哈希表中，则说明该链表是环形链表，否则就将该节点加入哈希表中。重复这一过程，直到我们遍历完整个链表即可。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        unordered_set<ListNode*> node;
        ListNode* cur=head;
        while(cur){
            //判断哈希表中是否已存在当前结点
            if(node.count(cur)) return true;
            //不存在 则加入哈希表
            node.insert(cur);
            cur=cur->next;
        } 
        return false;
    }
};
```

方法二：快慢指针

我们定义两个指针，一快一慢。慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 `head`，而快指针在位置 `head.next`。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表。否则快指针将到达链表尾部，该链表不为环形链表。

> 为什么我们要规定初始时慢指针在位置 head，快指针在位置 head.next，而不是两个指针都在位置 head？
>
> 当使用的是 while 循环，循环条件先于循环体。由于循环条件一定是判断快慢指针是否重合，如果我们将两个指针初始都置于 head，那么 while 循环就不会执行。因此，我们可以假想一个在 head 之前的虚拟节点，慢指针从虚拟节点移动一步到达 head，快指针从虚拟节点移动两步到达 head.next，这样我们就可以使用 while 循环了。
>
> 当然，我们也可以使用 do-while 循环。此时，我们就可以把快慢指针的初始值都置为 head。
>

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head==NULL || head->next==NULL) return false;
        ListNode* fast=head->next;
        ListNode* slow=head;
        while(slow!=fast){
            //因为fast比slow快 只需判断fast是否走到末尾
            if(fast==NULL || fast->next==NULL) return false;
            slow=slow->next;
            fast=fast->next->next;
        }
        return true;
    }
};
```

### 142.环形链表II ###

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

 

**示例 1：**

![img](assets\circularlinkedlisasfft.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![img](assets\circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![img](assets\circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

 

**提示：**

- 链表中节点的数目范围在范围 `[0, 104]` 内
- `-105 <= Node.val <= 105`
- `pos` 的值为 `-1` 或者链表中的一个有效索引

**思路：**

方法一：和上题方法一思路一样使用哈希表标记访问，若出现环只需返回当前节点即为环的第一个结点

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(head==NULL || head->next==NULL) return NULL;
        unordered_set<ListNode*> node;
        ListNode* cur=head;
        while(cur){
            if(node.count(cur)) return cur;//返回的结点即为环开始的点
            node.insert(cur);
            cur=cur->next;
        }
        return NULL;
    }
};
```

方法二：快慢指针

我们使用两个指针，fast 与 slow。它们起始都位于链表的头部。随后，slow 指针每次向后移动一个位置，而 fast 指针向后移动两个位置。如果链表中存在环，则 fast 指针最终将再次与 slow 指针在环中相遇。

如下图所示，设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与 fast 相遇。此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 `a+n(b+c)+b=a+(n+1)b+nc`。

<img src="assets\142_fig1.png" alt="fig1" style="zoom:67%;" />

根据题意，任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍。因此，我们有

`a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)`
有了 `a=c+(n−1)(b+c)` 的等量关系，我们会发现：**从相遇点到入环点的距离加上 n−1 圈的环长，恰好等于从链表头部到入环点的距离**。

因此，当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(head==NULL || head->next==NULL) return NULL;
        ListNode* fast=head,*slow=head;
        while(fast){
            if(fast->next==NULL) return NULL;
            slow=slow->next;
            fast=fast->next->next;
            //快慢指针相遇
            if(slow==fast){
                ListNode* ptr=head;
                while(ptr!=slow){
                    ptr=ptr->next;
                    slow=slow->next;
                }
                return ptr;
            }
        }
        return NULL;
    }
};
```

### 21.合并两个有序链表   ###

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 

**示例 1：**

![img](assets\merge_ex1.jpg)

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**示例 2：**

```
输入：l1 = [], l2 = []
输出：[]
```

**示例 3：**

```
输入：l1 = [], l2 = [0]
输出：[0]
```

 

**提示：**

- 两个链表的节点数目范围是 `[0, 50]`
- `-100 <= Node.val <= 100`
- `l1` 和 `l2` 均按 **非递减顺序** 排列

**思路**：

方法一：当 l1 和 l2 都不是空链表时，判断 l1 和 l2 哪一个链表的头节点的值更小，将较小值的节点添加到结果里，当一个节点被添加到结果里之后，将对应链表中的节点向后移一位。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(list1==nullptr && list2==nullptr) return nullptr;
        //维护两个指针
        ListNode* res=new ListNode(-1);//指向新链表的头结点的前一个位置
        ListNode* p=res;//用于在新链表中插入结点
        while(list1 && list2){
            if(list1->val < list2->val){
                p->next=list1;
                list1=list1->next;
            }else{
                p->next=list2;
                list2=list2->next;
            }
            p=p->next;
        }
        //最终至多只会有一个链表还有剩
        p->next= list1==nullptr?list2:list1;
        return res->next;
    }
};
```

方法二：递归

我们可以如下递归地定义两个链表里的 merge 操作（忽略边界情况，比如空链表等）：

- list1[0]+merge(list1[1:],list2)          list1[0]<list2[0]
- list2[0]+merge(list1,list2[1:])	  otherwise


也就是说，两个链表头部值较小的一个节点与剩下元素的 merge 操作结果合并。

如果 l1 或者 l2 一开始就是空链表 ，那么没有任何操作需要合并，所以我们只需要返回非空链表。否则，我们要判断 l1 和 l2 哪一个链表的头节点的值更小，然后递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(list1==nullptr) return list2;
        else if(list2==nullptr) return list1;
        else if(list1->val < list2->val){//当前结点中 l1的更小 
            list1->next=mergeTwoLists(list1->next,list2);//用l1下一个结点与剩下的l2继续合并
            return list1;//最终返回l1
        }else{
            list2->next=mergeTwoLists(list1,list2->next);//用l2下一个结点与剩下的l1继续合并
            return list2;//最终返回l2
        }
    }
};
```

### 2.两数相加

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

 

**示例 1：**

![img](assets\addtwonumber1.jpg)

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

**示例 2：**

```
输入：l1 = [0], l2 = [0]
输出：[0]
```

**示例 3：**

```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

 

**提示：**

- 每个链表中的节点数在范围 `[1, 100]` 内
- `0 <= Node.val <= 9`
- 题目数据保证列表表示的数字不含前导零

**思路：**

由于输入的两个链表都是逆序存储数字的位数的，因此两个链表中同一位置的数字可以直接相加。

我们同时遍历两个链表，逐位计算它们的和，并与当前位置的进位值相加。具体而言，如果当前两个链表处相应位置的数字为` n1,n2`，进位值为 `t`，则它们的和为 `n1+n2+t`；

其中，答案链表处相应位置的数字为` (n1+n2+t)mod10`，而新的进位值为`(n1+n2+t)/10`

如果两个链表的长度不同，则可以认为长度短的链表的后面有若干个 0 。

此外，如果链表遍历结束后，有 t>0，还需要在答案链表的后面附加一个节点，节点的值为 t。



```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 先造一个新链表
        ListNode* res = new ListNode(-1);
        ListNode* p = res; // 新链表的移动指针
        int t = 0;         // 设置进位值
        while (l1 || l2) {//只要有一个链表不为空就继续
            //遍历到某一链表为空时，设置其值为0，避免分别判断和处理
            int x= l1==nullptr?0:l1->val;
            int y= l2==nullptr?0:l2->val;
            int sum = x+y+ t;
            t = sum / 10;
            sum= sum % 10;
            p->next = new ListNode(sum);//两数之和结点
            p=p->next;
            //只要结点不为空 就后移
            if(l1) l1=l1->next;
            if(l2) l2=l2->next;
        }
        if (t) {//如果最后还有进位值
            p->next = new ListNode(t);
        }
        return res->next;
    }
};
```

### 19.删除链表的倒数第N个结点  ###

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

 

**示例 1：**

![img](assets\remove_ex1.jpg)

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**示例 2：**

```
输入：head = [1], n = 1
输出：[]
```

**示例 3：**

```
输入：head = [1,2], n = 1
输出：[1]
```

 

**提示：**

- 链表中结点的数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

**思路：**

方法一：**栈**

我们也可以在遍历链表的同时将所有节点依次入栈。根据栈「先进后出」的原则，我们弹出栈的第 n 个节点就是需要删除的节点，并且目前栈顶的节点就是待删除节点的前驱节点。这样一来，删除操作就变得十分方便了。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        //哑结点
        ListNode* dummy=new ListNode(0,head);
        //用栈存储结点
        stack<ListNode*> stk;
        ListNode* cur=dummy;
        while(cur){
            stk.push(cur);
            cur=cur->next;
        }
        //弹出n个结点
        for(int i=0;i<n;i++)
            stk.pop();
        ListNode* pre=stk.top();//即为要删除的结点的前一个结点
        pre->next=pre->next->next;
        ListNode* ans=dummy->next;
        delete dummy;
        return ans;
    }
};
```

方法二：如果我们要删除节点 y，我们需要知道节点 y 的前驱节点 x，并将 x 的指针指向 y 的后继节点。但由于头节点不存在前驱节点，因此我们需要在删除头节点时进行特殊判断。但如果我们添加了哑节点，那么头节点的前驱节点就是哑节点本身，此时我们就只需要考虑通用的情况即可。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //计算链表长度
    int getLength(ListNode* head){
        int length=0;
        while(head){
            length++;
            head=head->next;
        }
        return length;
    }
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        //设置一个哑结点 指向第一个结点
        ListNode* dummy=new ListNode(0,head);
        ListNode* cur=dummy;
        int length=getLength(head);
        for(int i=1;i<length-n+1;i++)
            cur=cur->next;
        cur->next=cur->next->next;
        ListNode* ans=dummy->next;
        delete dummy;//释放
        return ans;
    }
};
```

方法三：双指针-快慢指针

我们也可以在不预处理出链表的长度，以及使用常数空间的前提下解决本题。

由于我们需要找到倒数第 n 个节点，因此我们可以使用两个指针 first 和 second 同时对链表进行遍历，并且 first 比 second 超前 n 个节点。当 first 遍历到链表的末尾时，second 就恰好处于倒数第 n 个节点。

具体地，初始时 first 和 second 均指向头节点。我们首先使用 first 对链表进行遍历，遍历的次数为 n。此时，first 和 second 之间间隔了 n−1 个节点，即 first 比 second 超前了 n 个节点。

在这之后，我们同时使用 first 和 second 对链表进行遍历。当 first 遍历到链表的末尾（即 first 为空指针）时，second 恰好指向倒数第 n 个节点。

根据方法一和方法二，如果我们能够得到的是倒数第 n 个节点的前驱节点而不是倒数第 n 个节点的话，删除操作会更加方便。因此我们可以考虑在初始时将 second 指向哑节点，其余的操作步骤不变。这样一来，当 first 遍历到链表的末尾时，second 的下一个节点就是我们需要删除的节点。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 哑结点
        ListNode* dummy = new ListNode(0, head);
        ListNode* first = head;
        ListNode* second = dummy;
        // 先让first指针移动n次
        for (int i = 0; i < n; i++)
            first = first->next;
        // 然后first和second同时移动 直到first移动到末尾
        while (first) {
            first = first->next;
            second = second->next;
        }
        //此时second指向删除节点的前一个结点
        second->next = second->next->next;
        ListNode* ans = dummy->next;
        delete dummy;
        return ans;
    }
};
```

### 24.两两交换链表中的结点  ###

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

**示例 1：**

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

**示例 2：**

```
输入：head = []
输出：[]
```

**示例 3：**

```
输入：head = [1]
输出：[1]
```

 

**提示：**

- 链表中节点的数目在范围 `[0, 100]` 内
- `0 <= Node.val <= 100`

**思路**：

方法一：非递归解法

循环交换两个相邻结点（修改next指针），然后移步到下一对相邻结点（后移），此时需要处理前一次交换后第二个结点的next指针要指向当前一对相邻结点的第二个结点。继续循环

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head==nullptr) return nullptr;
        //定义两个指针指向相邻结点
        ListNode* p1=head;//第一个
        ListNode* p2=head->next;//第二个
        if(p2) head=p2;//结果的头结点 无非两种情况不变or指向旧链表的第二个结点
        ListNode* tmp=new ListNode(-1);//临时结点
        while(p1 && p2){
            //交换 修改p1和p2的next指针
            p1->next=p2->next;
            p2->next=p1;
            //指针后移
            tmp=p1;//先保存当前p1节点
            p1=p1->next;//p1后移
            if(p1){
                p2=p1->next;//p2后移
                tmp->next=p2;//此时原来p1下一个结点即为p2
            } 
            else break;//p1为空 直接结束
        }
        if(p1) tmp->next=p1;//如果循环后p1非空 即链表为奇数量
        return head;
    }
};
```

方法二：递归

递归的终止条件是链表中没有节点，或者链表中只有一个节点，此时无法进行交换。

如果链表中至少有两个节点，则在两两交换链表中的节点之后，原始链表的头节点变成新的链表的第二个节点，原始链表的第二个节点变成新的链表的头节点。链表中的其余节点的两两交换可以递归地实现。在对链表中的其余节点递归地两两交换之后，更新节点之间的指针关系，即可完成整个链表的两两交换。

用 `head` 表示原始链表的头节点，新的链表的第二个节点，用 `newHead `表示新的链表的头节点，原始链表的第二个节点，则原始链表中的其余节点的头节点是 `newHead.next`。令 `head.next = swapPairs(newHead.next)`，表示将其余节点进行两两交换，交换后的新的头节点为 `head` 的下一个节点。然后令 `newHead.next = head`，即完成了所有节点的交换。最后返回新的链表的头节点` newHead`。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head==nullptr || head->next==nullptr) 
            return head;//直接返回head
        ListNode* newHead=head->next;//新链表的头结点
        //新链表第二个结点的下一个结点
        head->next=swapPairs(newHead->next);//传入原链表第二个结点的下一个结点
        newHead->next=head;
        return newHead;//新链表头结点
        
    }
};
```

### 25.K个一组翻转链表  ###

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

 

**示例 1：**

```
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
```

**示例 2：**

```
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

 

**提示：**

- 链表中的节点数目为 `n`
- `1 <= k <= n <= 5000`
- `0 <= Node.val <= 1000`

**思路：**

1. 链表分区为已翻转部分+待翻转部分+未翻转部分

2. 每次翻转前，要确定翻转链表的范围，这个必须通过 `k`次循环来确定
3. 需记录翻转链表前驱和后继，方便翻转完成后把已翻转部分和未翻转部分连接起来
4. 初始需要两个变量 `pre` 和` end`，`pre `代表待翻转链表的前驱，`end `代表待翻转链表的末尾
5. 经过`k`次循环，`end `到达末尾，记录待翻转链表的后继 `next = end.next`
6. 翻转链表，然后将三部分链表连接起来，然后重置 `pre` 和 `end` 指针，然后进入下一次循环
7. 特殊情况，当翻转部分长度不足 `k` 时，在定位 `end` 完成后，`end==null`，已经到达末尾，说明题目已完成，直接返回即可

> [25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-nodes-in-k-group/solutions/10416/tu-jie-kge-yi-zu-fan-zhuan-lian-biao-by-user7208t/?envType=study-plan-v2&envId=top-100-liked)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //链表翻转
    ListNode* reverseL(ListNode *head){
        if(head==nullptr || head->next==nullptr) return head;
        //前驱结点
        ListNode* pre=nullptr;
        //当前节点
        ListNode* cur=head;
        //后继结点
        ListNode* next=nullptr;
        while(cur){
            next=cur->next;//保存后继结点
            cur->next=pre;
            pre=cur;
            cur=next;
        }
        return pre;

    }
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(head==nullptr || head->next==nullptr) return head;

        //定义哑结点
        ListNode* dummy=new ListNode(0);
        dummy->next=head;//next指向头结点
        //初始化pre和end都指向哑结点
        ListNode* pre=dummy;//pre指每次要翻转的链表的头结点的上一个节点
        ListNode* end=dummy;//end指每次要翻转的链表的尾节点

        while(end->next){//end->next即为下一个待翻转链表的头结点
            //先循环k次 找到当前要翻转链表的尾结点
            for(int i=0;i<k && end;i++){//主要要判断end是否为null 若为null意味着待翻转链表不满k个节点
                end=end->next;
            }
            if(end==nullptr) break;//直接退出
            //先记录end->next 即下一个待翻转链表的头结点
            ListNode* next=end->next;
            //然后断开
            end->next=nullptr;
            
            //开始翻转当前链表
            ListNode* start=pre->next;//头结点
            pre->next=reverseL(start);//翻转链表 同时返回翻转链表的头结点
            //断开的地方重新连接
            start->next=next;
            //将pre、end换成下次要翻转的链表的头结点的上一个节点。即start
            pre=start;
            end=start;
        }
        return dummy->next;
    }
};
```

### 138.随机链表的复制 ###

给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **[深拷贝](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)**。 深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点** 。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为 `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

 

**示例 1：**

![img](assets\e1.png)

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**示例 2：**

![img](assets\e2.png)

```
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```

**示例 3：**

![img](assets\e3.png)

```
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

 

**提示：**

- `0 <= n <= 1000`
- `-104 <= Node.val <= 104`
- `Node.random` 为 `null` 或指向链表中的节点。

**思路：**[138. 随机链表的复制 - 力扣（LeetCode）](https://leetcode.cn/problems/copy-list-with-random-pointer/solutions/2361362/138-fu-zhi-dai-sui-ji-zhi-zhen-de-lian-b-6jeo/?envType=study-plan-v2&envId=top-100-liked)

方法一：哈希表

利用哈希表的查询特点，考虑构建 **原链表节点** 和 **新链表对应节点** 的键值对映射关系，再遍历构建新链表各节点的 `next` 和 `random` 引用指向即可。

算法流程：

1.若头节点 `head` 为空节点，直接返回 null 。

2.**初始化：** 哈希表 `dic` ， 节点 `cur` 指向头节点。

3.复制链表：

1. 建立新节点，并向 `dic` 添加键值对 `(原 cur 节点, 新 cur 节点）` 。
2. `cur` 遍历至原链表下一节点。

4.构建新链表的引用指向：

1. 构建新节点的 `next` 和 `random` 引用指向。
2. `cur` 遍历至原链表下一节点。

5.**返回值：** 新链表的头节点 `dic[cur]` 。

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head==NULL) return NULL;
        Node* cur=head;//指向头结点
        //构建哈希表 键为原链表结点，值为新链表结点
        unordered_map<Node*,Node*> map;
        //建立map的键值映射"原结点-->新结点"
        while(cur){
            map[cur]=new Node(cur->val);//创建新结点
            cur=cur->next;
        }
        cur=head;
        //构建新链表的next和random的指向
        while(cur){
            map[cur]->next=map[cur->next];
            map[cur]->random=map[cur->random];
            cur=cur->next;
        }
        return map[head];//返回新链表的头结点
    }
};
```

### 148.排序链表 ###

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

 

**示例 1：**

![img](assets\sort_list_1.jpg)

```
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```

**示例 2：**

![img](assets\sort_list_2.jpg)

```
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

**示例 3：**

```
输入：head = []
输出：[]
```

 

**提示：**

- 链表中节点的数目在范围 `[0, 5 * 104]` 内
- `-105 <= Node.val <= 105`

**思路：**

[148. 排序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-list/solutions/13728/sort-list-gui-bing-pai-xu-lian-biao-by-jyd/?envType=study-plan-v2&envId=top-100-liked)

方法一：归并排序（递归法）

- **分割 cut 环节**： 找到当前链表 中点，并从 中点 将链表断开（以便在下次递归 `cut` 时，链表片段拥有正确边界）；
  - 我们使用` fast`,`slow `快慢双指针法，奇数个节点找到中点，偶数个节点找到中心左边的节点。
  - 找到中点` slow` 后，执行 `slow.next = null `将链表切断。
  - 递归分割时，输入当前链表左端点 `head` 和中心节点 `slow` 的下一个节点 `tmp`(因为链表是从 `slow `切断的)。
  - **cut 递归终止条件**： 当 `head.next == None `时，说明只有一个节点了，直接返回此节点。
- **合并 merge 环节**： 将两个排序链表合并，转化为一个排序链表。
  - 双指针法合并，建立辅助 `ListNode h `作为头部。
  - 设置两指针 `left, right `分别指向两链表头部，比较两指针处节点值大小，由小到大加入合并链表头部，指针交替前进，直至添加完两个链表。
  - 返回辅助`ListNode h` 作为头部的下个节点` h.next。`
  - 时间复杂度` O(l + r)`，`l, r` 分别代表两个链表长度。
  - 当题目输入的 `head == null `时，直接返回 `null`。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if(head==nullptr || head->next==nullptr) return head;
        //快慢指针 找中点分割
        ListNode *fast=head->next,*slow=head;
        while(fast && fast->next){
            fast=fast->next->next;
            slow=slow->next;
        }
        //找到中点 断开两边链表
        ListNode* tmp=slow->next;
        slow->next=nullptr;
        //递归两边分割的的链表
        ListNode* left=sortList(head);
        ListNode* right=sortList(tmp);

        //合并
        ListNode* h=new ListNode(0);
        ListNode* res=h;
        while(left && right){
            if(left->val < right->val){//较小值放入结果链表
                h->next=left;
                left=left->next;
            }else{
                h->next=right;
                right=right->next;
            }
            h=h->next;//后移
        } 
        h->next= left?left:right;//最后将还有剩余的链表接入结果链表
        return res->next;//返回结果链表头结点
    }
};
```

### 23.合并K个升序链表 ###

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

**示例 1：**

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**

```
输入：lists = []
输出：[]
```

**示例 3：**

```
输入：lists = [[]]
输出：[]
```

 

**提示：**

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` 按 **升序** 排列
- `lists[i].length` 的总和不超过 `10^4`

**思路：**

方法一：使用优先队列合并(构建**小根堆**)

我们需要维护当前每个链表没有被合并的元素的最前面一个，k 个链表就最多有 k 个满足这样条件的元素，每次在这些元素里面选取 val 属性最小的元素合并到答案中。在选取最小元素的时候，我们可以用优先队列来优化这个过程。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //重载优先队列中的元素比较规则 构造小根堆
    struct comp{
        bool operator()(ListNode* a,ListNode* b){//返回true时，a排在b的后面
            return a->val > b->val;
        }
    };
    //优先队列
    priority_queue<ListNode*,vector<ListNode*>,comp> q;

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        for(auto node:lists){//先将所有链表的头结点加入堆中
            if(node) q.push(node);
        }
        ListNode* head=new ListNode();
        ListNode* tail=head;
        while(!q.empty()){//队列不为空时
            //弹出队头元素 即为当前所有加入结点中的最小值
            ListNode* node=q.top();
            q.pop();
            tail->next=node;
            tail=tail->next;
            if(node->next) q.push(node->next);
        }
        return head->next;
    }

};
```

方法二：顺序合并

每次两两合并

用一个变量 `ans`来维护以及合并的链表，第` i `次循环把第 `i `个链表和 `ans` 合并，答案保存到 `ans` 中。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //两两合并
    ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
        if((a==nullptr) || (b==nullptr)) return a?a:b;//若有一个为空
        ListNode head, *tail = &head, *aPtr = a, *bPtr = b;
        while (aPtr && bPtr) {
            if (aPtr->val < bPtr->val) {
                tail->next = aPtr; aPtr = aPtr->next;
            } else {
                tail->next = bPtr; bPtr = bPtr->next;
            }
            tail = tail->next;
        }
        tail->next = (aPtr ? aPtr : bPtr);
        return head.next;
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *ans = nullptr;
        for (size_t i = 0; i < lists.size(); ++i) {
            ans = mergeTwoLists(ans, lists[i]);
        }
        return ans;
    }

};
```



### 146.LRU缓存 ###

请你设计并实现一个满足 [LRU (最近最少使用) 缓存](https://baike.baidu.com/item/LRU) 约束的数据结构。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value` ；如果不存在，则向缓存中插入该组 `key-value` 。如果插入操作导致关键字数量超过 `capacity` ，则应该 **逐出** 最久未使用的关键字。

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

 

**示例：**

```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

 

**提示：**

- `1 <= capacity <= 3000`
- `0 <= key <= 10000`
- `0 <= value <= 105`
- 最多调用 `2 * 105` 次 `get` 和 `put`

**思路：**哈希表 + 双向链表

LRU 缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。

- 双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。

- 哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。


这样以来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在 O(1) 的时间内完成 get 或者 put 操作。具体的方法如下：

- 对于` get `操作，首先判断 `key `是否存在：

  - 如果` key` 不存在，则返回 `−1`；

  - 如果` key` 存在，则 `key `对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。

- 对于 `put `操作，首先判断 `key `是否存在：

  - 如果` key `不存在，使用 `key` 和` value `创建一个新的节点，在双向链表的头部添加该节点，并将 `key `和该节点添加进哈希表中。然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；

  - 如果 `key` 存在，则与` get` 操作类似，先通过哈希表定位，再将对应的节点的值更新为 `value`，并将该节点移到双向链表的头部。


上述各项操作中，访问哈希表的时间复杂度为 `O(1)`，在双向链表的头部添加节点、在双向链表的尾部删除节点的复杂度也为` O(1)`。而将一个节点移到双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，都可以在 `O(1) `时间内完成。

注：

在双向链表的实现中，使用一个伪头部（哨兵节点dummy）next指向链表的第一个结点，pre指向最后一个结点

```c++
//创建一个双向链表
class Node{
public:
    int key,value;
    Node *pre,*next;
    Node(int k=0,int v=0):key(k),value(v){}
};
class LRUCache {
private:
    int capacity;
    Node* dummy;//哨兵结点指向第一个结点
    //哈希表 键为key，值为链表结点
    unordered_map<int,Node*> mp;

    //删除一个结点
    void remove(Node* x){
        x->pre->next=x->next;
        x->next->pre=x->pre;
    }
    //链表头部插入一个结点
    void insertHead(Node* x){
        x->next=dummy->next;
        dummy->next->pre=x;
        dummy->next=x;
        x->pre=dummy;
    }
    //得到一个键为key的结点
    Node* getNode(int key){
        auto it=mp.find(key);
        if(it==mp.end()) return nullptr;//不存在该结点
        //如果存在该结点
        auto node=it->second;//得到该结点
        //删除该节点
        remove(node);
        //再将该结点加入链表头部
        insertHead(node);
        return node;
    }

public:
    LRUCache(int capacity):capacity(capacity),dummy(new Node()) {
        //初始化 哨兵结点的前后指针都指向自身
        dummy->pre=dummy;
        dummy->next=dummy;
    }
    
    int get(int key) {
        auto node=getNode(key);//是否存在该结点
        return node?node->value:-1;
    }
    
    void put(int key, int value) {
        auto node=getNode(key);//是否存在该节点
        if(node){//存在该结点
            node->value=value;
            return ;
        }
        //不存在该结点
        node=new Node(key,value);//创建
        mp[key]=node;//加入哈希表
        insertHead(node);//放在链表头部
        if(mp.size()>capacity){//若超出容量
            auto back_node=dummy->pre;//得到尾结点
            remove(back_node);//从链表中删除尾节点
            mp.erase(back_node->key);//同时从哈希表中删除
            delete back_node;//释放内存
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

## 二叉树

### 94.二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。

 

**示例 1：**

![img](assets\inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [1]
输出：[1]
```

 

**提示：**

- 树中节点数目在范围 `[0, 100]` 内
- `-100 <= Node.val <= 100`

**思路：**

方法一：递归

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void inorderTree(TreeNode* root,vector<int> &res){
        if(root==nullptr) return ;
        inorderTree(root->left,res);
        res.push_back(root->val);
        inorderTree(root->right,res);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        inorderTree(root,res);
        return res;
    }
};
```

方法二：迭代方式（递归方法最为直观易懂，但考虑到效率，我们通常不推荐使用递归。）

和递归是等价的，只是需要我们手动模拟栈。

左子树先入栈，然后出栈，输出根节点，然后右子树入栈，再出栈

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        while(root || !st.empty()){//根节点或栈不为空
            //左子树入栈
            while(root){
                st.push(root);
                root=root->left;
            }
            //左子树全部入栈 再出栈
            root=st.top();
            st.pop();
            //根节点输出
            res.push_back(root->val);
            //处理右子树入栈
            root=root->right;
        }
        return res;
    }
};
```

方法三：迭代的另一版本[94. 二叉树的中序遍历 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-inorder-traversal/solutions/25220/yan-se-biao-ji-fa-yi-chong-tong-yong-qie-jian-ming/?envType=study-plan-v2&envId=top-100-liked)

**兼具栈迭代方法的高效，又像递归方法一样简洁易懂，更重要的是，这种方法对于前序、中序、后序遍历，能够写出完全一致的代码**。

其核心思想如下：

- 使用颜色标记节点的状态，新节点为白色，已访问的节点为灰色。

- 如果遇到的节点为白色，则将其标记为灰色，然后将其右子节点、自身、左子节点依次入栈。
- 如果遇到的节点为灰色，则将节点的值输出。

更通俗理解：

遍历一棵树时，按照递归的思路理解，分为 **进入** 和 **回溯** 两个阶段，用栈模拟可以理解为 "两次入栈"：

- 第一次入栈时是以当前节点为根节点的 **整棵子树入栈**；
- 通过栈中序遍历该子树，就要对其进行 **展开**，第二次入栈代表展开。对于任意一棵树，中序遍历都是先递归左子树，因此需要按照 **右子树-中节点-左子树** 的顺序入栈展开；

两次入栈就同样对应着两次出栈：

- 第一次出栈是展开前将代表子树的栈顶节点出栈；
- 第二次出栈是展开后栈顶的**中节点**加入遍历序列；

具体地说，采用变量 flag 标记节点两次入栈的过程，flag = 0 代表第一次入栈，flag = 1 代表第二次入栈。首先根节点标记为 0 入栈，迭代取出栈顶节点时：

- 当栈顶节点的 flag = 0 时，代表子树递归进入的过程，先将栈顶节点出栈，然后按照 右子树-中节点-左子树 的顺序将该子树展开入栈，其中右子树和左子树标记为 0，中节点标记为 1
- 当 flag = 1 时，代表递归回溯的过程，将栈顶节点加入到中序遍历序列

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        //分别为结点 及其 标记值 0 or 1
        stack<pair<TreeNode*,int>> st;
        //先将当前结点加入栈 标记为0
        st.push({root,0});
        
        while(!st.empty()){//栈不为空
            //弹出栈顶节点
            auto [node,type]=st.top();
            st.pop();
            if(node==nullptr) continue;

            if(type==0){//如果栈顶结点标记值为0 代表第一次入栈 “递归”
                st.push({node->right,0});//右结点入栈
                st.push({node,1});//当前结点再入栈 标记为1
                st.push({node->left,0});//左结点入栈
            }else{//标记值为1 输出  “回溯”
                res.push_back(node->val);
            }
        }
        return res;
    }
};
```

### 104.二叉树的最大深度

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

 

**示例 1：**

![img](assets\tmp-tree.jpg)

 

```
输入：root = [3,9,20,null,null,15,7]
输出：3
```

**示例 2：**

```
输入：root = [1,null,2]
输出：2
```

 

**提示：**

- 树中节点的数量在 `[0, 104]` 区间内。
- `-100 <= Node.val <= 100`

**思路：**

方法一：深度优先搜索DFS

分别递归左右子树，取大值再加1

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root==nullptr) return 0;
        return max(maxDepth(root->left),maxDepth(root->right))+1;
    }
};
```

方法二：广度优先遍历BFS（层序遍历）

关键点： 每遍历一层，则计数器 +1 ，直到遍历完成，则可得到树的深度。

算法解析：

1. 特例处理： 当 root 为空，直接返回 深度 0 。
2. 初始化： 队列 queue （加入根节点 root ），计数器 res = 0。
3. 循环遍历： 当 queue 为空时跳出。
   1. 初始化一个空列表 tmp ，用于临时存储下一层节点。
   2. 遍历队列： 遍历 queue 中的各节点 node ，并将其左子节点和右子节点加入 tmp。
   3. 更新队列： 执行 queue = tmp ，将下一层节点赋值给 queue。
   4. 统计层数： 执行 res += 1 ，代表层数加 1。
4. 返回值： 返回 res 即可。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root==nullptr) return 0;
        vector<TreeNode*> que;//队列
        que.push_back(root);
        int res=0;
        while(!que.empty()){//队列不为空时
            vector<TreeNode*> tmp;//存储下一层的结点
            for(auto* node:que){//遍历队列 将各结点的子节点加入临时队列
                if(node->left) tmp.push_back(node->left);
                if(node->right) tmp.push_back(node->right);
            }
            que=tmp;
            res++;//一层过后深度加1
        }
        return res;
    }
};
```

### 226.翻转二叉树 ###

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\invert1-tree.jpg)

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

**示例 2：**

![img](D:\桌面\za\算法\assets\invert2-tree.jpg)

```
输入：root = [2,1,3]
输出：[2,3,1]
```

**示例 3：**

```
输入：root = []
输出：[]
```

 

**提示：**

- 树中节点数目范围在 `[0, 100]` 内
- `-100 <= Node.val <= 100`

**思路**：递归

我们从根节点开始，递归地对树进行遍历，并从叶子节点先开始翻转。如果当前遍历到的节点 root 的左右两棵子树都已经翻转，那么我们只需要交换两棵子树的位置，即可完成以 root 为根节点的整棵子树的翻转。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left),
 * right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr)
            return nullptr;
        //注意递归翻转后返回结点
        TreeNode* left = invertTree(root->left);
        TreeNode* right = invertTree(root->right);
        //再进行左右交换
        root->left = right;
        root->right = left;
        return root;
    }
};
```

### 101.对称二叉树  ###

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\1698026966-JDYPDU-image.png)

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

![img](D:\桌面\za\算法\assets\1698027008-nPFLbM-image.png)

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

 

**提示：**

- 树中节点数目在范围 `[1, 1000]` 内
- `-100 <= Node.val <= 100`

**思路：**

对称二叉树定义： 对于树中 任意两个对称节点 L 和 R ，一定有：

- `L.val = R.val `：即此两对称节点值相等。
- `L.left.val = R.right.val` ：即 L 的 左子节点 和 R 的 右子节点 对称。
- ` L.right.val = R.left.val `：即 L 的 右子节点 和 R 的 左子节点 对称。

根据以上规律，考虑从顶至底递归，判断每对左右节点是否对称，从而判断树是否为对称二叉树。

函数` isSymmetric(root) `：

- 特例处理： 若根节点` root` 为空，则直接返回` true `。
- 返回值： 即` recur(root.left, root.right)` ;

函数 `recur(L, R) `：

- 终止条件：

  - 当 `L `和` R `同时越过叶节点： 此树从顶至底的节点都对称，因此返回 true 。

  - 当` L `或` R `中只有一个越过叶节点： 此树不对称，因此返回 false 。

  - 当节点` L` 值 =≠节点 `R` 值： 此树不对称，因此返回 false 。

- 递推工作：

  - 判断两节点 `L.left `和 `R.right `是否对称，即` recur(L.left, R.right) `。

  - 判断两节点 `L.right `和` R.left `是否对称，即 `recur(L.right, R.left) `。

- 返回值： 两对节点都对称时，才是对称树，因此用与逻辑符 `&& `连接。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool recur(TreeNode* L,TreeNode* R){
        //同时为空
        if(L==nullptr && R==nullptr) return true;
        //出现一个为空 一个非空 或者两者都非空但值不等
        if(L==nullptr || R==nullptr || L->val != R->val) return false;
        //递归处理左右子树
        return recur(L->left,R->right) && recur(L->right,R->left);
    }
    bool isSymmetric(TreeNode* root) {
        return recur(root->left,root->right);
    }
};
```

### 543.二叉树的直径 ###

给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\diamtree.jpg)

```
输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
```

**示例 2：**

```
输入：root = [1,2]
输出：1
```

 

**提示：**

- 树中节点数目在范围 `[1, 104]` 内
- `-100 <= Node.val <= 100`

**思路**：

假设我们知道对于该节点的左儿子向下遍历经过最多的节点数` L `（即以左儿子为根的子树的深度） 和其右儿子向下遍历经过最多的节点数` R `（即以右儿子为根的子树的深度），那么以**该节点为中间点的路径经过节点数的最大值即为 `L+R+1` 。**

我们记节点` node` 为中间点的路径经过节点数的最大值为 `d_node`，那么**二叉树的直径就是所有节点 `d_node`的最大值减一。**

最后的算法流程为：我们定义一个递归函数 `depth(node)` 计算` d_node`，函数返回该节点为根的子树的深度。先递归调用左儿子和右儿子求得它们为根的子树的深度` L` 和` R `，则该节点为根的子树的深度即为`max(L,R)+1`

该节点的 `d_node`值为`L+R+1`

递归搜索每个节点并设一个全局变量 `ans` 记录` d_node`的最大值，最后返回` ans-1` 即为树的直径。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int res;
    //递归求以结点为根的最大深度 同时更新res
    int depth(TreeNode* root){
        if(root==nullptr) return 0;
        //分别求左右子树的深度
        int L=depth(root->left);
        int R=depth(root->right);
        res=max(res,L+R+1);//更新经过该点的最多节点数
        return max(L,R)+1;//返回以该结点为根的最大深度
    }
    int diameterOfBinaryTree(TreeNode* root) {
        res=1;
        depth(root);
        return res-1;
    }
};
```

### 102.二叉树的层序遍历

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\tree1.jpg)

```
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

**示例 2：**

```
输入：root = [1]
输出：[[1]]
```

**示例 3：**

```
输入：root = []
输出：[]
```

 

**提示：**

- 树中节点数目在范围 `[0, 2000]` 内
- `-1000 <= Node.val <= 1000`

**思路：广度优先遍历BFS**

- 首先根元素入队
- 当队列不为空的时候
  - 求当前队列的长度 `s_i`
  - 依次从队列中取 `s_i`个元素进行拓展，然后进入下一次迭代

它和普通广度优先搜索的区别在于，普通广度优先搜索每次只取一个元素拓展，而这里每次取` s_i`个元素。在上述过程中的第` i` 次迭代就得到了二叉树的第 `i `层的 `s_i`个元素。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(root==nullptr) return res;

        //创建一个队列
        queue<TreeNode*> q;
        q.push(root);//根节点入队
        while(!q.empty()){//队列不为空时 一层一层处理
            //先得到队列的长度
            int length=q.size();
            res.push_back(vector<int>());//在结果中为当前层创建一个新数组
            //处理当前层 取出队列的所有元素
            for(int i=0;i<length;i++){
                //队头出队
                auto node=q.front();
                q.pop();
                //值加入该层
                res.back().push_back(node->val);
                //扩展 将当前队列结点的子节点都加入队列 即下一层结点
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
        }
        return res;
    }
};
```

### 108.将有序数组转换为二叉搜索树   ###

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡二叉搜索树。



**示例 1：**

![img](D:\桌面\za\算法\assets\btree1.jpg)

```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/02/18/btree.jpg)

```
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

 

**提示：**

- `1 <= nums.length <= 104`
- `-104 <= nums[i] <= 104`
- `nums` 按 **严格递增** 顺序排列

**思路**：

二叉搜索树的中序遍历是升序序列，题目给定的数组是按照升序排序的有序数组，因此可以确保数组是二叉搜索树的中序遍历序列。

如果没有要求二叉搜索树的高度平衡，则任何一个数字都可以作为二叉搜索树的根节点，因此可能的二叉搜索树有多个。

直观地看，我们可以**选择中间数字作为二叉搜索树的根节点，这样分给左右子树的数字个数相同或只相差 1，可以使得树保持平衡**。如果数组长度是奇数，则根节点的选择是唯一的，如果数组长度是偶数，则可以选择中间位置左边的数字作为根节点或者选择中间位置右边的数字作为根节点，选择不同的数字作为根节点则创建的平衡二叉搜索树也是不同的。

确定平衡二叉搜索树的根节点之后，其余的数字分别位于平衡二叉搜索树的左子树和右子树中，左子树和右子树分别也是平衡二叉搜索树，因此可以通过递归的方式创建平衡二叉搜索树。

递归的基准情形是平衡二叉搜索树不包含任何数字，此时平衡二叉搜索树为空。 在给定中序遍历序列数组的情况下，每一个子树中的数字在数组中一定是连续的，因此可以通过数组下标范围确定子树包含的数字，下标范围记为` [left,right]`。 对于整个中序遍历序列，下标范围从 `left=0` 到 `right=nums.length−1`。 当 `left>right` 时，平衡二叉搜索树为空。

方法一：中序遍历，总是选择中间位置左边的数字作为根节点 选择中间位置左边的数字作为根节点，则根节点的下标为 mid=(left+right)/2

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //递归构建二叉搜索树（平衡）
    TreeNode* helper(vector<int> & nums,int left,int right){
        if(left>right) return nullptr;//终止条件
        //每次选择中间位置左边的数作为根节点
        int mid=(left+right)/2;
        TreeNode* root=new TreeNode(nums[mid]);
        //递归构建左子树
        root->left=helper(nums,left,mid-1);
        //递归构建右子树
        root->right=helper(nums,mid+1,right);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return helper(nums,0,nums.size()-1);
    }
};
```

方法二：与方法一相反，总是选择中间位置左边的数字作为根节点 选择中间位置左边的数字作为根节点，则根节点的下标为 mid=(left+right+1)/2

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //递归构建二叉搜索树（平衡）
    TreeNode* helper(vector<int> & nums,int left,int right){
        if(left>right) return nullptr;//终止条件
        //每次选择中间位置左边的数作为根节点
        int mid=(left+right+1)/2;
        TreeNode* root=new TreeNode(nums[mid]);
        //递归构建左子树
        root->left=helper(nums,left,mid-1);
        //递归构建右子树
        root->right=helper(nums,mid+1,right);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return helper(nums,0,nums.size()-1);
    }
};
```

### 98.验证二叉搜索树 ###

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。

- 节点的右子树只包含 **大于** 当前节点的数。

- 所有左子树和右子树自身必须也是二叉搜索树。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\tree564511.jpg)

```
输入：root = [2,1,3]
输出：true
```

**示例 2：**

![img](D:\桌面\za\算法\assets\tre5564e2.jpg)

```
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
```

 

**提示：**

- 树中节点数目范围在`[1, 104]` 内
- `-231 <= Node.val <= 231 - 1`

**思路：**

方法一：递归

如果该二叉树的左子树不为空，则左子树上所有节点的值均小于它的根节点的值； 若它的右子树不空，则右子树上所有节点的值均大于它的根节点的值；它的左右子树也为二叉搜索树。

这启示我们设计一个递归函数 `helper(root, lower, upper) `来递归判断，函数表示考虑以` root `为根的子树，判断子树中所有节点的值是否都在 `(l,r) `的范围内（注意是开区间）。如果` root `节点的值` val `不在 `(l,r) `的范围内说明不满足条件直接返回，否则我们要继续递归调用检查它的左右子树是否满足，如果都满足才说明这是一棵二叉搜索树。

那么根据二叉搜索树的性质，在递归调用左子树时，我们需要把上界` upper` 改为 `root.val`，即调用 `helper(root.left, lower, root.val)`，因为左子树里所有节点的值均小于它的根节点的值。同理递归调用右子树时，我们需要把下界` lower` 改为` root.val`，即调用 `helper(root.right, root.val, upper)`。

函数递归调用的入口为` helper(root, -inf, +inf)， inf `表示一个无穷大的值。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //递归 判断节点的值是否在制定范围内
    bool helper(TreeNode* root,long long l,long long r){//使用long long型 因为测试数据可能超过整数范围
    if(root==nullptr) return true;
    if(root->val <=l || root->val>=r) return false;//超出范围
    return helper(root->left,l,root->val) && helper(root->right,root->val,r);//注意左右子树上下界

    }
    bool isValidBST(TreeNode* root) {
        return helper(root,LONG_MIN,LONG_MAX);
    }
};
```

方法二：中序遍历

二叉搜索树「中序遍历」得到的值构成的序列一定是升序的，这启示我们在中序遍历的时候实时检查当前节点的值是否大于前一个中序遍历到的节点的值即可。如果均大于说明这个序列是升序的，整棵树是二叉搜索树，否则不是，下面的代码我们使用栈来模拟中序遍历的过程。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
//============================中序遍历递归版=======================
class Solution {
public:
    long long pre=(long long )INT_MIN-1;//前一个结点值
    bool isValidBST(TreeNode* root) {
        if(root==nullptr) return true;
        //递归左子树
        bool l=isValidBST(root->left);
        //判断当前结点值是否大于前一个结点值
        bool tmp= pre < root->val;
        if(tmp) pre=root->val;//更新值
        //递归右子树
        bool r=isValidBST(root->right);
        return l && tmp && r;
    }
};

//============================中序遍历迭代版=======================
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> st;
        //前一个结点的值
        long long pre=(long long)INT_MIN-1;
        
        while(!st.empty() || root){
            //先根节点及其左子树左节点入队
            while(root){
                st.push(root);
                root=root->left;
            }
            //弹出栈顶元素
            root=st.top();
            st.pop();
            //如果当前节点的值小于等于前一个结点的值 返回false
            if(root->val <= pre) return false;
            //否则 更新前一个结点的值
            pre=root->val;
            root=root->right;//遍历右节点
        }
        return true;
    }
};
```

### 230.二叉搜索树中第K小的元素

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\kthtree1.jpg)

```
输入：root = [3,1,4,null,2], k = 1
输出：1
```

**示例 2：**

![img](D:\桌面\za\算法\assets\kthtree2.jpg)

```
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```

 

 

**提示：**

- 树中的节点数为 `n` 。
- `1 <= k <= n <= 104`
- `0 <= Node.val <= 104`

**思路：**中序遍历

- 递归遍历时计数，统计当前节点的序号。
- 递归到第 k 个节点时，应记录结果 res 。
- 记录结果后，后续的遍历即失去意义，应提前返回。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int res,k;//存储结果及计数
    void helper(TreeNode* root){
        //k等于0时提前返回
        if(root==nullptr || k==0) return;
        helper(root->left);
        if(--k==0) res=root->val; //找到目标值
        helper(root->right);
        return ;
    }
    int kthSmallest(TreeNode* root, int k) {
        this->k=k;
        helper(root);
        return res;
    }
};
```

### 199.二叉树的右视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

 

**示例 1:**

![img](D:\桌面\za\算法\assets\tree.jpg)

```
输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
```

**示例 2:**

```
输入: [1,null,3]
输出: [1,3]
```

**示例 3:**

```
输入: []
输出: []
```

 

**提示:**

- 二叉树的节点个数的范围是 `[0,100]`
- `-100 <= Node.val <= 100` 

思路：层序遍历

将每层的最后一个元素加入结果

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if(root==nullptr) return res;
        //创建一个队列 层序遍历处理
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){//处理每一层
            int length=q.size();
            TreeNode* node=nullptr;
            for(int i=0;i<length;i++){//扩展 加入下一层元素
                node=q.front();
                q.pop();
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
            //向结果数组中插入该层的最后一个元素
            res.push_back(node->val);
        }
        return res;
    }
};
```

### 114.二叉树展开为链表  ###

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 **先序遍历**顺序相同。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\flaten.jpg)

```
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [0]
输出：[0]
```

 

**提示：**

- 树中结点数在范围 `[0, 2000]` 内
- `-100 <= Node.val <= 100`

**思路：**

1. 将左子树插入到右子树的地方
2. 将原来的右子树接到左子树的最右边节点
3. 考虑新的右子树的根节点，一直重复上边的过程，直到新的右子树为 null

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    void flatten(TreeNode* root) {
        if(root==nullptr) return;
        flatten(root->left);//递归处理左子树
        flatten(root->right);//递归右子树
        TreeNode* l=root->left;//得到根节点左子树
        if(l){//左子树不为空
            while(l->right) l=l->right;//找到左子树最右边的结点
            //将根节点右子树接到左子树最右边的结点的右边
            l->right=root->right;
            //左子树接到根节点的右边
            root->right=root->left;
            //根节点左边置空
            root->left=nullptr;
        }
    }
};
```

### 105.从前序与中序遍历序列构造二叉树  ###

给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

 

**示例 1:**

![img](D:\桌面\za\算法\assets\tre564561e.jpg)

```
输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]
```

**示例 2:**

```
输入: preorder = [-1], inorder = [-1]
输出: [-1]
```

 

**提示:**

- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` 和 `inorder` 均 **无重复** 元素
- `inorder` 均出现在 `preorder`
- `preorder` **保证** 为二叉树的前序遍历序列
- `inorder` **保证** 为二叉树的中序遍历序列

**思路：**

方法一：递归+哈希表

对于任意一颗树而言，前序遍历的形式总是：

**[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]**。

即根节点总是前序遍历中的第一个节点。而中序遍历的形式总是：

**[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]**

只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。

这样以来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，我们就可以递归地对构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。

注：

在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。**我们可以考虑使用哈希表来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。**在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 O(1) 的时间对根节点进行定位了。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //哈希表 键存结点的值 值存结点在中序遍历中的位置
    unordered_map<int,int> mp;

    //递归构建树 设置四个变量表示对前序遍历和中序遍历处理到的区间位置
    TreeNode* myBuildTree(vector<int> &preorder,vector<int> &inorder,int pre_l,int pre_r,int in_l,int in_r){
        if(pre_l>pre_r) return nullptr;//终止条件

        //前序遍历的第一个结点就是根节点
        int pre_root=pre_l;//得到索引
        //以此在中序遍历中定位到根节点
        int in_root=mp[preorder[pre_root]];

        //创建出根节点
        TreeNode* root=new TreeNode(preorder[pre_root]);
        //得到左子树的节点个数 
        int left_size=in_root-in_l;

        //递归构造左子树，并连接到根节点
        /*
        先序遍历中「从 左边界+1 开始的 left_size」个元素
        就对应了
        中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        */
        root->left=myBuildTree(preorder,inorder,pre_l+1,pre_l+left_size,in_l,in_root-1);
        
        //同理构造右子树
        root->right=myBuildTree(preorder,inorder,pre_l+left_size+1,pre_r,in_root+1,in_r);
        return root;
    }


    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n=preorder.size();
        //对中序遍历序列构造哈希映射，快速定位根节点
        for(int i=0;i<n;i++){
            mp[inorder[i]]=i;
        }
        return myBuildTree(preorder,inorder,0,n-1,0,n-1);
    }
};
```

### 437.路径总和III ###

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\pathsum3-1-tree.jpg)

```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
```

**示例 2：**

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
```

 

**提示:**

- 二叉树的节点个数的范围是 `[0,1000]`
- `-109 <= Node.val <= 109` 
- `-1000 <= targetSum <= 1000` 

**思路：**

我们首先想到的解法是穷举所有的可能，我们访问每一个节点 `node`，检测以 `node` 为起始节点且向下延深的路径有多少种。我们递归遍历每一个节点的所有可能的路径，然后将这些路径数目加起来即为返回结果。

- 我们首先定义` rootSum(p,val) `表示以节点 `p `为起点向下且满足路径总和为 `val `的路径数目。我们对二叉树上每个节点 p 求出 `rootSum(p,targetSum)`，然后对这些路径数目求和即为返回结果。

- 我们对节点 p 求` rootSum(p,targetSum) `时，以当前节点 p 为目标路径的起点递归向下进行搜索。假设当前的节点 `p` 的值为 `val`，我们对左子树和右子树进行递归搜索，对节点 `p` 的左孩子节点 `pl`求出` rootSum(pl,targetSum−val)`，以及对右孩子节点 `pr`求出 `rootSum(pr,targetSum−val)`。节点 `p` 的 `rootSum(p,targetSum) `即等于 `rootSum(pl ,targetSum−val) `与` rootSum(pr,targetSum−val)` 之和，同时我们还需要判断一下当前节点 `p` 的值是否刚好等于 `targetSum`。
- 我们采用递归遍历二叉树的每个节点 `p`，对节点` p `求 `rootSum(p,val)`，然后将每个节点所有求的值进行相加求和返回。


```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //求以某节点为根节点的符合条件的路径数
    int rootSum(TreeNode* root,int targetSum){
        if(root==nullptr) return 0;
        int res=0;//记录路径数
        //当前结点恰好等于目标值
        if(root->val==targetSum) res++; 

        //分别递归左右子树求路径数 然后相加
        res+=rootSum(root->left,targetSum-root->val);
        res+=rootSum(root->right,targetSum-root->val);
        return res;
    }
    
    int pathSum(TreeNode* root, int targetSum) {
        if(root==nullptr) return 0;
        //遍历二叉树的每个结点 然后求和
        int res=rootSum(root,targetSum);
        res+=pathSum(root->left,targetSum);
        res+=pathSum(root->right,targetSum);
        return res;
    }
};
```

方法二：前缀和+哈希表

[437. 路径总和 III - 力扣（LeetCode）](https://leetcode.cn/problems/path-sum-iii/solutions/2784856/zuo-fa-he-560-ti-shi-yi-yang-de-pythonja-fmzo/?envType=study-plan-v2&envId=top-100-liked)

### 236.二叉树的最近公共祖先 ###

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

 

**示例 1：**

![img](D:\桌面\za\算法\assets\binarytree.png)

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

**示例 2：**

![img](D:\桌面\za\算法\assets\binarytree.png)

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```

**示例 3：**

```
输入：root = [1,2], p = 1, q = 2
输出：1
```

 

**提示：**

- 树中节点数目在范围 `[2, 105]` 内。
- `-109 <= Node.val <= 109`
- 所有 `Node.val` `互不相同` 。
- `p != q`
- `p` 和 `q` 均存在于给定的二叉树中。

**思路：**

若 root 是 p,q 的 最近公共祖先 ，则只可能为以下情况之一：

- p 和 q 在 root 的子树中，且分列 root 的 异侧（即分别在左、右子树中）；
- p=root ，且 q 在 root 的左或右子树中；
- q=root ，且 p 在 root 的左或右子树中；

考虑通过递归对二叉树进行先序遍历，当遇到节点 p 或 q 时返回。从底至顶回溯，当节点 p,q 在节点 root 的异侧时，节点 root 即为最近公共祖先，则向上返回 root 。

**递归解析：**

1. **终止条件**：

   - 当越过叶节点，则直接返回 null ；

   - 当 root 等于 p,q ，则直接返回 root ；

2. **递推工作**：

   - 开启递归左子节点，返回值记为 left ；

   - 开启递归右子节点，返回值记为 right ；

3. **返回值**： 根据 left 和 right ，可展开为四种情况；

   - 当 left 和 right 同时为空 ：说明 root 的左 / 右子树中都不包含 p,q ，返回 null ；

   - 当 left 和 right 同时不为空 ：说明 p,q 分列在 root 的 异侧 （分别在 左 / 右子树），因此 root 为最近公共祖先，返回 root ；

   - 当 left 为空 ，right 不为空 ：p,q 都不在 root 的左子树中，直接返回 right 。具体可分为两种情况：

     - p,q 其中一个在 root 的 右子树 中，此时 right 指向 p（假设为 p ）；

     - p,q 两节点都在 root 的 右子树 中，此时的 right 指向 最近公共祖先节点 ；

   - 当 left 不为空 ， right 为空 ：与情况 3. 同理；

观察发现， 情况 1. 可合并至 3. 和 4. 内

[236. 二叉树的最近公共祖先 - 力扣（LeetCode）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/solutions/240096/236-er-cha-shu-de-zui-jin-gong-gong-zu-xian-hou-xu/?envType=study-plan-v2&envId=top-100-liked)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //只要当前根节点为空 或是q，p的任意一个就返回 因为再深入就会越过其中一个点
        if(root==NULL || root==q || root==p) return root;
        //否则 继续递归左右子树找q，p
        TreeNode* left=lowestCommonAncestor(root->left,p,q);
        TreeNode* right=lowestCommonAncestor(root->right,p,q);
        //如果还是没找到q,p，那就没有
        if(left==NULL && right==NULL) return NULL;
        //左子树没有q，p，而右子树结果不为空，返回右子树的结果
        if(left==NULL) return right;
        //同理右子树没有q，p，返回左子树结果
        if(right==NULL) return left;

        //左右子树都找到了q，p，说明q，p在root的两侧，此时最近公共祖先就是root
        return root;
    }
};
```

### 124.二叉树中的最大路径和 ###

二叉树中的 **路径** 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 **至多出现一次** 。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

 

**示例 1：**

![img](D:\桌面\za\408\25 md笔记\assets\exx1.jpg)

```
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
```

**示例 2：**

![img](D:\桌面\za\408\25 md笔记\assets\exx2.jpg)

```
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

 

**提示：**

- 树中节点数目范围是 `[1, 3 * 104]`
- `-1000 <= Node.val <= 1000`

**思路：**

思路同543.二叉树的直径，只是将求以某节点为根的最大深度改为求以某节点为根的最大结点之和

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int res;
    //递归求以某节点为根结点的最大结点和
    int getNodeSum(TreeNode* root){
        if(root==nullptr) return 0;
        int l=getNodeSum(root->left);//左子树的最大结点和
        int r=getNodeSum(root->right);//右子树的最大结点和
        res=max(res,l+r+root->val);//注意加上根节点值
        return max(max(l,r)+root->val,0);//注意这里还要和0比较 因为节点存在负值
    }

    int maxPathSum(TreeNode* root) {
        res=INT_MIN;//初始化为最小值
        getNodeSum(root);
        return res;
    }
};
```

## 图论

### 200.岛屿的数量

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

 

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` 的值为 `'0'` 或 `'1'`

**思路：**

岛屿类问题的通用解法、DFS 遍历框架:[200. 岛屿数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-islands/solutions/211211/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/?envType=study-plan-v2&envId=top-100-liked)

解法实际上就是遍历矩阵，遍历到为陆地的点，岛屿数就+1，且对其周围所有的陆地标记一下，表示已有归属（防止重复遍历）

方法一：深度遍历优先DFS

- 目标是找到矩阵中 “岛屿的数量” ，上下左右相连的 1 都被认为是连续岛屿。

- `dfs`方法： 设目前指针指向一个岛屿中的某一点 `(i, j)`，寻找包括此点的岛屿边界。
  - 从` (i, j)` 向此点的上下左右 `(i+1,j)`,`(i-1,j)`,`(i,j+1)`,`(i,j-1)` 做深度搜索。
  - 终止条件：
    - `(i, j)` 越过矩阵边界;
    - `grid[i][j] == 0`，代表此分支已越过岛屿边界。
  - 搜索岛屿的同时，执行 `grid[i][j] = '0'`，即将岛屿所有节点删除，以免之后重复搜索相同岛屿。
- 主循环：
  - 遍历整个矩阵，当遇到` grid[i][j] == '1'` 时，从此点开始做深度优先搜索 `dfs`，岛屿数 `count + 1 `且在深度优先搜索中删除此岛屿。
- 最终返回岛屿数 `count `即可。

```c++
class Solution {
public:
    //对某点进行深度优先遍历
    void dfs(vector<vector<char>>& grid,int i,int j){
        //越界 或者 当前点为海（或访问过）
        if(i<0 || j<0 || i>=grid.size() || j>=grid[0].size() || grid[i][j]== '0') return ;
        grid[i][j]='0';//标记为海 也即意味着访问过
        dfs(grid,i+1,j);
        dfs(grid,i-1,j);
        dfs(grid,i,j+1);
        dfs(grid,i,j-1);
    }
    int numIslands(vector<vector<char>>& grid) {
        int count=0;//记录岛屿数
        //遍历整个矩阵
        for(int i=0;i<grid.size();i++)
            for(int j=0;j<grid[0].size();j++){
                if(grid[i][j]=='1'){//为陆地时
                    dfs(grid,i,j);
                    count++;//岛屿数+1
                }
            }
            return count;
    }
};
```

方法二：广度优先遍历

- 借用一个队列 queue，判断队列首部节点 (i, j) 是否未越界且为 1：

  - 若是则置零（删除岛屿节点），并将此节点上下左右节点 (i+1,j),(i-1,j),(i,j+1),(i,j-1) 加入队列；
  - 若不是则跳过此节点；
- 循环 pop 队列首节点，直到整个队列为空，此时已经遍历完此岛屿。

```c#
class Solution {
public:
    //对某点进行广度优先遍历
    void bfs(vector<vector<char>>& grid,int i,int j){
        //创建一个队列 记录当前点
        queue<pair<int,int>> q;
        q.push({i,j});//入队
        while(!q.empty()){
            //队头出队
            auto tmp=q.front();
            q.pop();
            int r=tmp.first,c=tmp.second;
            if(r<0 || c<0 || r>=grid.size() || c>= grid[0].size() || grid[r][c]=='0') continue;//判断越界 或 已访问过（为海）跳过该点
            grid[r][c]='0';//标记
            //四个方位入队
            q.push({r+1,c});
            q.push({r-1,c});
            q.push({r,c+1});
            q.push({r,c-1});
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        int count=0;//记录岛屿数
        //遍历整个矩阵
        for(int i=0;i<grid.size();i++)
            for(int j=0;j<grid[0].size();j++){
                if(grid[i][j]=='1'){//为陆地时
                    bfs(grid,i,j);
                    count++;//岛屿数+1
                }
            }
            return count;
    }
};
```



### 994.腐烂的橘子

在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回 *直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`* 。



**示例 1：**

**![img](D:\桌面\za\408\25 md笔记\assets\oranges.png)**

```
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4
```

**示例 2：**

```
输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个方向上。
```

**示例 3：**

```
输入：grid = [[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
```

 

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 10`
- `grid[i][j]` 仅为 `0`、`1` 或 `2`

**思路：**

- 一开始，我们找出所有腐烂的橘子，将它们放入队列，作为第 0 层的结点。

- 然后进行 BFS 遍历，每个结点的相邻结点可能是上、下、左、右四个方向的结点，注意判断结点位于网格边界的特殊情况。
- 由于可能存在无法被污染的橘子，我们需要记录新鲜橘子的数量。在 BFS 中，每遍历到一个橘子（污染了一个橘子），就将新鲜橘子的数量减一。如果 BFS 结束后这个数量仍未减为零，说明存在无法被污染的橘子。

```c++
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int M = grid.size();
        int N = grid[0].size();
        queue<pair<int,int>> q;
        vector<vector<int>> dir = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // 四个方向移动

        int count = 0; // count 表示新鲜橘子的数量
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == 1) {
                    count++;
                } else if (grid[r][c] == 2) {
                    q.push({r, c}); // 腐烂的入队
                }
            }
        }

        int round = 0; // round 表示分钟数
        while (count > 0 && !q.empty()) { // 注意条件 count不能少，否则会多计算
            round++;
            int n = q.size();
            for (int i = 0; i < n; i++) {
                auto tmp=q.front();
                q.pop();
                for (int k = 0; k < 4; k++) {
                    int cr = tmp.first + dir[k][0];
                    int cc = tmp.second + dir[k][1];
                    if (cr >= 0 && cr < M && cc >= 0 && cc < N && grid[cr][cc] == 1) {
                        grid[cr][cc] = 2; // 开始腐烂
                        count--;
                        q.push({cr, cc}); // 添加新元素
                    }
                }
            }
        }
        if (count > 0) {
            return -1;
        } else {
            return round;
        }
    }
};
```

### 207.课程表

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```

**示例 2：**

```
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```

 

**提示：**

- `1 <= numCourses <= 2000`
- `0 <= prerequisites.length <= 5000`
- `prerequisites[i].length == 2`
- `0 <= ai, bi < numCourses`
- `prerequisites[i]` 中的所有课程对 **互不相同**

**思路**：本题可约化为： 课程安排图是否是 **有向无环图(DAG)**。即课程间规定了前置条件，但不能构成任何环路，否则课程前置条件将不成立。

思路是通过 **拓扑排序** 判断此课程安排图是否是 **有向无环图(DAG)** 。

>  拓扑排序原理： 对 DAG 的顶点进行排序，使得对每一条有向边 (u,v)，均有 u（在排序记录中）比 v 先出现。亦可理解为对某点 v 而言，只有当 v 的所有源点均出现了，v 才能出现。

通过课程前置条件列表 prerequisites 可以得到课程安排图的 **邻接表 adjacency**，以降低算法时间复杂度，以下两种方法都会用到邻接表。

方法一：广度优先遍历-拓扑排序

1. 统计课程安排图中每个节点的入度，生成 **入度表** `indegrees`。
2. 借助一个队列 `queue`，将所有入度为 0 的节点入队。
3. 当` queue `非空时，依次将队首节点出队，在课程安排图中删除此节点 `pre`：
   1. 并不是真正从邻接表中删除此节点 `pre`，而是将此节点对应所有邻接节点` cur `的入度 −1，即` indegrees[cur] -= 1`。
   2. 当入度 −1后邻接节点 `cur` 的入度为 0，说明 `cur `所有的前驱节点已经被 “删除”，此时将 cur 入队。
4. 在每次 pre 出队时，执行` numCourses--`；
   1. 若整个课程安排图是有向无环图（即可以安排），则所有节点一定都入队并出队过，即完成拓扑排序。换个角度说，若课程安排图中存在环，一定有节点的入度始终不为 0。
   2. 因此，拓扑排序出队次数等于课程个数，返回 `numCourses == 0 `判断课程是否可以成功安排。

```c++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        //记录每个点的入度
        vector<int> indegress(numCourses);
        //邻接表
        vector<vector<int>> adjacency(numCourses);
        //创建一个队列存储入度为0的结点
        queue<int> q;
        //初始化入度表和邻接表
        for(auto pq:prerequisites){
            indegress[pq[0]]++;
            adjacency[pq[1]].push_back(pq[0]);
        }
        //遍历入度表 将入度为0的课程加入队列
        for(int i=0;i<numCourses;i++)
            if(!indegress[i]) q.push(i);
        //广度优先遍历
        while(!q.empty()){
            //取出队头元素
            auto pre=q.front();
            q.pop();
            numCourses--;//课程数-1
            for(auto cur:adjacency[pre])//处理该课程的出边结点
                //如果所有出边结点的入度-1后为0 加入队列
                if(--indegress[cur]==0) q.push(cur);
        }
        return numCourses==0;
    }
};
```

方法二：深度优先遍历判断是否有环

1. 借助一个标志列表 `flags`，用于判断每个节点` i `（课程）的状态：

   1. 未被 DFS 访问：`i == 0`；
   2. 已**被其他节点启动的 DFS** 访问：`i == -1`；
   3. 已**被当前节点启动的 DFS** 访问：`i == 1`。
2. 对 `numCourses `个节点依次执行` DFS`，判断每个节点起步` DFS `是否存在环，若存在环直接返回 $False$。`DFS `流程；
   1. 终止条件：
      1. 当` flag[i] == -1`，说明当前访问节点已被其他节点启动的 DFS 访问，无需再重复搜索，直接返回 $True$。
      2. 当 `flag[i] == 1`，说明在本轮 DFS 搜索中节点 i 被第 2 次访问，即 **课程安排图有环** ，直接返回$False$。
   2. 将当前访问节点 `i `对应 `flag[i] `置 1，即标记其被本轮 DFS 访问过；
   3. 递归访问当前节点` i `的所有邻接节点` j`，当发现环直接返回 $False$；
   4. 当前节点所有邻接节点已被遍历，并没有发现环，则将当前节点 `flag` 置为 −1 并返回 $True$。
3. 若整个图 DFS 结束并未发现环，返回 $True$。

[207. 课程表 - 力扣（LeetCode）](https://leetcode.cn/problems/course-schedule/solutions/18806/course-schedule-tuo-bu-pai-xu-bfsdfsliang-chong-fa/?envType=study-plan-v2&envId=top-100-liked)

### 208.实现Trie树

**[Trie](https://baike.baidu.com/item/字典树/9825209?fr=aladdin)**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

 

**示例：**

```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

 

**提示：**

- `1 <= word.length, prefix.length <= 2000`
- `word` 和 `prefix` 仅由小写英文字母组成
- `insert`、`search` 和 `startsWith` 调用次数 **总计** 不超过 `3 * 104` 次

**思路：**

Trie树，又称字典树，是用来**高效存储和查找字符串集合**的一种数据结构查找时，可以**高效的查找某个字符串**是否在Trie树中出现过，并且可以查找出现了多少次

**主要性质：**

- **根节点不包含字符，除根节点外的每一个子节点都包含一个字符。**
- 从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。

- 每个节点的所有子节点包含的字符互不相同。

- 从第一字符开始有连续重复的字符只占用一个节点，比如上面的catch和cat中重复的单词cat只占用了一个节点。

**插入**
描述：向 `Trie` 中插入一个单词 `word`

实现：这个操作和构建链表很像。首先从根结点的子结点开始与` word` 第一个字符进行匹配，一直匹配到前缀链上没有对应的字符，这时开始不断开辟新的结点，直到插入完` word `的最后一个字符，同时还要将最后一个结点`isEnd = true`;，表示它是一个单词的末尾。

**查找**
描述：查找` Trie` 中是否存在单词 `word`

实现：从根结点的子结点开始，一直向下匹配即可，如果出现结点值为空就返回` false`，如果匹配到了最后一个字符，那我们只需判断 `node->isEnd`即可。

**前缀匹配**
描述：判断 `Trie` 中是否有以` prefix `为前缀的单词

实现：和 `search` 操作类似，只是不需要判断最后一个字符结点的`isEnd`，因为既然能匹配到最后一个字符，那后面一定有单词是以它为前缀的。

```c++
class Trie {
private://定义Trie结构--即每个结点存储的信息
    bool isEnd;//表示是否存在以当前节点为结尾的字符串
    Trie* next[26];//当前结点存在哪些子节点(26个字母)
public:
    
    Trie() {//初始化
         isEnd=false;
         memset(next,0,sizeof(next));//置0
    }
    
    void insert(string word) {
        Trie* node=this;//指针初始指向根节点
        for(auto c:word){
            if(node->next[c-'a']==NULL)//当前结点下面不存在该字符
                node->next[c-'a']=new Trie;//则在下面创建该字符
            node=node->next[c-'a'];//指针下移
        }
        node->isEnd=true;//存在以最后一个字符结尾的字符串
    }
    
    bool search(string word) {
        Trie* node=this;
        for(auto c:word){
            node=node->next[c-'a'];
            if(node==NULL) return false;//出现不存在的情况直接返回
        }
        return node->isEnd;//最后返回isEnd 看是否存在以这个字符串最后一个字符结尾的标记
    }
    
    bool startsWith(string prefix) {
        Trie* node=this;
        for(auto c:prefix){
            node=node->next[c-'a'];
            if(node==NULL) return false;//出现不存在的情况直接返回
        }
        return true;//否则返回true
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

## 回溯

**回溯详述**：[46. 全排列 - 力扣（LeetCode）](https://leetcode.cn/problems/permutations/solutions/9914/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/?envType=study-plan-v2&envId=top-100-liked)

**回溯法** 采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：

- 找到一个可能存在的正确的答案；

- 在尝试了所有可能的分步方法后宣告该问题没有答案。



### 46.全排列

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```
输入：nums = [1]
输出：[[1]]
```

 

**提示：**

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有整数 **互不相同**

**思路：**

利用深度优先搜索

- 每次搜索确定一个位置的数字，然后回溯判断是否有其它可能，**回溯后要恢复现场**
- 设置一个结果数组`path`，存储每次确定放置的数字。`dfs`递归参数设置为`u`，表示要判断的第几个数字，若`u==n`表示找到一个序列输出，然后回溯（注意u从0开始，`u==n`时就意味着已经放置好了n个数）。
  - 设置一个bool数组`st[]`，`st[i]=true`表示数字i已经放置，否则未放置
  - 若未放置，则将i放置在当前位置`path[u]`，且设置对应数组`st[i]=true`
  - 然后递归到下一个为止`dfs(u+1)`，继续判断放置
  - 到达最后一个位置输出一次结果后，**回溯，恢复现场，设置`st[i]=false`**，然后继续循环判断这个位置放另一个数

```c++
class Solution {
private:
    vector<vector<int>> res;//存储结果
    vector<int> path;//存储每一组结果
public:

    //u表示已经放好了几个数字 
    void dfs(int u,vector<bool> &st,vector<int> nums){
        if(u==nums.size()){//已经得到一组结果
            res.emplace_back(path);//存储
            return ;
        }
        //否则继续填充一组结果
        for(int i=0;i<nums.size();i++){
            if(!st[i]){//当前数字还未被放置
                path.emplace_back(nums[i]);//放置
                st[i]=true;
                dfs(u+1,st,nums);//递归继续放置下一个数
                //回溯 要恢复现场
                path.pop_back();//拿出
                st[i]=false;
            }
        }
        return ;
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> st(nums.size(),false);//表示当前数字是否已经放好
        dfs(0,st,nums);//初始放置0个数
        return res;
    }
};
```

### 78.子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

> 数组的 **子集** 是从数组中选择一些元素（可能为空）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

 

**提示：**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有元素 **互不相同**

方法一：位运算

使用`n`位的二进制数表示某个数字是否在一个子集中，如`nums=[1,2,3]`，用`3`位二进制位。`000`表示子集`[]`，`001`表示子集`[3]`，....，`111`表示子集`[1,2,3]`。即用`n`个二进制位，第`i`位表示数字`a_i`是否在子集中，则对于长度为`n`的数组，需要枚举十进制数$x$范围$0到2^n-1$。

对于一个十进制数x，求其第k位的二进制数：`x>>k & 1`  右移k位再和1做与运算

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        for(int i=0;i< (1<<nums.size());i++){//枚举0~2^n-1
            vector<int> t;//一个子集
            for(int j=0;j<nums.size();j++){//表示右移位数
                int k= (i>>j) & 1;//右移j位后最低位的数字(0或1)
                if(k==1) t.push_back(nums[j]);//加入子集
            }
            res.push_back(t);//一组结果
        }
        return res;
    }
};
```

方法二：回溯

`dfs(cur,n)` 参数表示当前位置是 cur，原序列总长度为 n。

原序列的每个位置在答案序列中的状态**有被选中和不被选中两种**，我们用 `t `数组存放已经被选出的数字。

在进入` dfs(cur,n)` 之前` [0,cur−1] `位置的状态是确定的，而 `[cur,n−1] `内位置的状态是不确定的，`dfs(cur,n)` 需要确定 `cur `位置的状态，然后求解子问题` dfs(cur+1,n)`。

对于` cur` 位置，我们需要考虑 `a[cur] `取或者不取

- 如果取，我们需要把` a[cur]` 放入一个临时的答案数组 `t`，再执行 `dfs(cur+1,n)`，执行结束后需要对 `t` 进行回溯；
- 如果不取，则直接执行 `dfs(cur+1,n)`。

在整个递归调用的过程中，cur 是从小到大递增的，当 cur 增加到 n 的时候，记录答案并终止递归。

```c++
class Solution {
public:
    vector<int> t;
    vector<vector<int>> res;
    void dfs(int cur,vector<int> nums){
        if(cur==nums.size()){//放好了一组结果
            res.push_back(t);
            return ;
        }
        //对于当前数cur
        //选择
        t.push_back(nums[cur]);
        dfs(cur+1,nums);//递归选下一个
        //回溯 即不选择当前数
        t.pop_back();
        dfs(cur+1,nums);
    }
    vector<vector<int>> subsets(vector<int>& nums) {
            dfs(0,nums);
            return res;
    }
};
```

方法三：动态规划

可以这么表示，`dp[i]`表示前i个数的解集，`dp[i] = dp[i - 1] + collections(i)`。其中，`collections(i)`表示把`dp[i-1]`的所有子集都加上第`i`个数形成的子集。

【具体操作】

因为`nums`大小不为0，故解集中一定有空集。令解集一开始只有空集，然后遍历`nums`，每遍历一个数字，拷贝解集中的所有子集，将该数字与这些拷贝组成新的子集再放入解集中即可。时间复杂度为$O(n^2)$。

1. 例如`[1,2,3]`，一开始解集为`[[]]`，表示只有一个空集。
2. 遍历到`1`时，依次拷贝解集中所有子集，只有`[]`，把`1`加入拷贝的子集中得到`[1]`，然后加回解集中。此时解集为`[[], [1]]`。
3. 遍历到`2`时，依次拷贝解集中所有子集，有`[], [1]`，把`2`加入拷贝的子集得到`[2], [1, 2]`，然后加回解集中。此时解集为`[[], [1], [2], [1, 2]]`。
4. 遍历到`3`时，依次拷贝解集中所有子集，有`[], [1], [2], [1, 2]`，把`3`加入拷贝的子集得到`[3], [1, 3], [2, 3], [1, 2, 3]`，然后加回解集中。此时解集为`[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]`。

```c#
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) { 
        vector<vector<int>> res;//结果集
        res.push_back(vector<int>());//先加入一个空集
        for(int i=0;i<nums.size();i++){
            int size=res.size();//前一次的子集数
            for(int j=0;j<size;j++){
                vector<int> t=res[j];//拷贝前一轮的每个子集
                t.push_back(nums[i]);//每个子集加入当前元素
                res.push_back(t);//更新结果集
            }
        }
        return res; 
    }
};
```

### 17.电话号码的字母组合

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)

 

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```
输入：digits = ""
输出：[]
```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]
```

 

**提示：**

- `0 <= digits.length <= 4`
- `digits[i]` 是范围 `['2', '9']` 的一个数字。

**思路：**

方法一：回溯

```c++
class Solution {
public:
    string tmp;
    vector<string> res;
    vector<string> board = {"",    "",    "abc",  "def", "ghi",
                            "jkl", "mno", "pqrs", "tuv", "wxyz"};
    void DFS(int pos, string digits) {
        if (pos == digits.size()) {//存放好一组结果
            res.push_back(tmp);
            return;
        }
        int num = digits[pos] - '0';
        for (int i = 0; i < board[num].size(); i++) {
            tmp.push_back(board[num][i]);
            DFS(pos + 1, digits);//组合下一个字母
            tmp.pop_back();//回溯
        }
    }
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0)
            return res;
        DFS(0, digits);
        return res;
    }
};
```

方法二：队列法

 先放入a、b、c，然后把a提出，放入ad、ae、af，再把b提出，放入bd、be、bf······

```c++
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> res{""};//初始化加入一个空字符串 此时大小为1
        if (digits.empty()){
            res.erase(res.begin());//擦除空字符串
            return res;//返回
        } 
        //哈希表存储数字和对应字符串
        unordered_map<char, string> mp {
            {'2', "abc"}, 
            {'3', "def"}, 
            {'4', "ghi"}, 
            {'5', "jkl"},
            {'6', "mno"}, 
            {'7', "pqrs"}, 
            {'8', "tuv"}, 
            {'9', "wxyz"}
        };

        string tmp;//临时变量
        /*
        先放入a、b、c，然后把a提出，放入ad、ae、af，
        再把b提出，放入bd、be、bf······
        */
        for(char& i:digits){//遍历digits
            int n=res.size();
            for(int j=0;j<n;j++){//循环res的大小次数
                tmp=res[0];//把第一个元素提出来 依次是a b c
                res.erase(res.begin());//提出来后就删掉第一个元素
                for(auto& k:mp.at(i)){//返回数字i对应的字符串
                    res.push_back(tmp+k);//组合 放入
                }
            }
        }
        return res;
    }
};
```

### 39.组合总和

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

 

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

**示例 2：**

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```
输入: candidates = [2], target = 1
输出: []
```

 

**提示：**

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- `candidates` 的所有元素 **互不相同**
- `1 <= target <= 40`

**思路**：

例如，输入集合 {3,4,5} 和目标整数 9 ，解为 {3,3,3},{4,5} 。需要注意两点：

- 输入集合中的元素可以被无限次重复选取。

- 子集是不区分元素顺序的，比如 {4,5} 和 {5,4} 是同一个子集。

输入数组 [3,4,5] 和目标元素 9 ，输出结果为 [3,3,3],[4,5],[5,4] 。**虽然成功找出了所有和为 9 的子集，但其中存在重复的子集 [4,5] 和 [5,4] 。**

这是因为搜索过程是区分选择顺序的，然而子集不区分选择顺序。

**重复子集剪枝**：

**我们考虑在搜索过程中通过剪枝进行去重**。

1. 第一轮和第二轮分别选择 3 , 4 ，会生成包含这两个元素的所有子集，记为 [3,4,⋯] 。

2. **若第一轮选择 4 ，则第二轮应该跳过 3** ，因为该选择产生的子集 [4,3,⋯] 和 1. 中生成的子集完全重复。

为实现该剪枝，我们初始化变量 start ，用于指示遍历起点。当做出选择 $x_i$ 后，设定下一轮从索引 `i`开始遍历。这样做就可以让选择序列满足 $i_1≤i_2 ≤⋯≤i_m $，从而保证子集唯一。

除此之外：

- 在开启搜索前，先将数组` nums` 排序。在遍历所有选择时，当子集和超过` target `时直接结束循环，因为后边的元素更大，其子集和都一定会超过 `target `。
- 省去元素和变量` total`，通过在` target `上执行减法来统计元素和，当 `target `等于 0 时记录解。

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> t;//一组结果
    //实现剪枝(去掉重复子集) 使用start变量指示下一个遍历起点
    void dfs(int target,int start,vector<int> candidates){
        if(target==0){//子集和为target 一组解
            res.push_back(t);
            return ;
        }
        //剪枝：从start开始，避免重复子集
        for(int i=start;i<candidates.size();i++){
            if(candidates[i]>target) break;//直接结束 后面不会有结果
            //尝试寻找结果
            t.push_back(candidates[i]);
            dfs(target-candidates[i],i,candidates);
            t.pop_back();
        }

    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        //优化：先对数组排序，在遍历过程中若子集和已经超过target直接结束
        sort(candidates.begin(),candidates.end());
        int start=0;//遍历起点
        dfs(target,start,candidates);
        return res;
    }
};
```



### 22.括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

 

**示例 1：**

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

**示例 2：**

```
输入：n = 1
输出：["()"]
```

 

**提示：**

- `1 <= n <= 8`

**思路：**

我们可以只在序列仍然保持有效时才添加 ‘(’ 或 ‘)’。我们可以通过跟踪到目前为止放置的左括号和右括号的数目来做到这一点，

如果左括号数量不大于 n，我们可以放一个左括号。然后左括号放完，如果右括号数量小于左括号的数量，我们可以放一个右括号。

```c#
class Solution {
public:
    //n表示括号对数量 left表示已放左括号数量 right表示已放右括号数量
    void dfs(string &cur,vector<string> &res,int n,int left,int right){
        if(left==n && right==n){//若左右括号数量已满足要求
            res.push_back(cur);
            return ;
        }
        //否则从左括号开始放置 直至数量满足
        if(left<n){
            cur.push_back('(');
            dfs(cur,res,n,left+1,right);//左括号数量+1
            cur.pop_back();//回溯 恢复现场
        }
        //此时左括号已放置好 开始放置右括号
        if(right<left){//右括号数量比左括号少时
            cur.push_back(')');
            dfs(cur,res,n,left,right+1);
            cur.pop_back();
        }

    }
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string cur;
        dfs(cur,res,n,0,0);
        return res;
    }
};
```

### 79.单词搜索

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 

**示例 1：**

![img](D:\桌面\za\算法\assets\word2.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

![img](D:\桌面\za\算法\assets\word-1.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
```

**示例 3：**

![img](D:\桌面\za\算法\assets\word3.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

 

**提示：**

- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` 和 `word` 仅由大小写英文字母组成

**思路：**深度优先搜索（DFS）+ 剪枝

- 深度优先搜索： 即暴力法遍历矩阵中所有字符串可能性。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。

- 剪枝： 在搜索中，遇到“这条路不可能和目标字符串匹配成功”的情况，例如当前矩阵元素和目标字符不匹配、或此元素已被访问，则应立即返回，从而避免不必要的搜索分支。

- **递归参数**： 当前元素在矩阵 `board `中的行列索引` i `和` j `，当前目标字符在 `word `中的索引` k` 。
- **终止条件**：
  - 返回 false ： 
    - (1) 行或列索引越界 
    - (2) 当前矩阵元素与目标字符不同 
    -  (3) 当前矩阵元素已访问过 （ (3) 可合并至 (2) ） 。
  - 返回 true ： `k = len(word) - 1` ，即字符串` word `已全部匹配。
- **递推工作**：
  - 标记当前矩阵元素： 将 `board[i][j] `修改为 空字符 ' ' ，代表此元素已访问过，防止之后搜索时重复访问。
  - 搜索下一单元格： 朝当前元素的 上、下、左、右 四个方向开启下层递归，使用 或 连接 （代表只需找到一条可行路径就直接返回，不再做后续 DFS ），并记录结果至 `res `。
  - 还原当前矩阵元素： 将` board[i][j] `元素还原至初始值，即 `word[k] `。
- **返回值**： 返回布尔量` res `，代表是否搜索到目标字符串。

> 使用空字符（Java/C++: '\0' ）做标记是为了防止标记字符与矩阵原有字符重复。当存在重复时，此算法会将矩阵原有字符认作标记字符，从而出现错误。

```c++
class Solution {
public:
    bool dfs(vector<vector<char>>& board, string word,int i,int j,int k){
        //剪枝：判断越界 以及 表格当前位置字母与word的当前字母是否相等 否则立即返回false
        if(i>=board.size() || i<0 || j>=board[0].size() || j<0 ||board[i][j]!=word[k]) return false;
        //终止条件
        if(k==word.size()-1) return true;
        board[i][j]='\0';//标记 表示此处已被访问
        //四个方向递归
        bool res=dfs(board,word,i+1,j,k+1)
                || dfs(board,word,i-1,j,k+1)
                || dfs(board,word,i,j+1,k+1)
                || dfs(board,word,i,j-1,k+1);
        board[i][j]=word[k];//恢复现场
        return res;
    }
    bool exist(vector<vector<char>>& board, string word) {
        for(int i=0;i<board.size();i++)
            for(int j=0;j<board[0].size();j++){
                //遍历 沿一个方向挑选适合的路径
                if(dfs(board,word,i,j,0)) return true;
            }
        return false;
    }
};
```

### 131.分割回文串

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串**。返回 `s` 所有可能的分割方案。

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

 

**提示：**

- `1 <= s.length <= 16`
- `s` 仅由小写英文字母组成

**思路：**回溯 + 动态规划预处理

由于需要求出字符串 `s` 的所有分割方案，因此我们考虑使用搜索 + 回溯的方法枚举所有可能的分割方法并进行判断。

假设我们当前搜索到字符串的第` i` 个字符，且` s[0..i−1] `位置的所有字符已经被分割成若干个回文串，并且分割结果被放入了答案数组` ans` 中，那么我们就需要枚举下一个回文串的右边界` j`，使得 `s[i..j] `是一个回文串。

因此，我们可以从` i` 开始，从小到大依次枚举` j`。对于当前枚举的` j `值，我们使用双指针的方法判断` s[i..j] `是否为回文串：如果 `s[i..j]` 是回文串，那么就将其加入答案数组` ans` 中，并以` j+1 `作为新的 `i` 进行下一层搜索，并在未来的回溯时将` s[i..j]` 从 `ans `中移除。

如果我们已经搜索完了字符串的最后一个字符，那么就找到了一种满足要求的分割方法。

对于判断`s[i..j]` 是否是回文串，常规的方法是使用双指针分别指向 `i` 和 `j`，每次判断两个指针指向的字符是否相同，直到两个指针相遇。然而这种方法会产生重复计算

采用动态规划预处理得到`s[i..j]` 是否是回文串，设 `f(i,j) `表示 `s[i..j] `是否为回文串，那么有状态转移方程：`f[i][j]=s[i]==s[j] && f[i+1][j-1]`

```c++
class Solution {
private:
vector<vector<int>> f;//f[i][j]表示i到j的字符串是否为回文串
vector<vector<string>> res;
vector<string> ans;
public:
    //分割字符串 组成回文串
    void dfs(string &s,int i){   
        if(i==s.size()){
            res.push_back(ans);
            return ;
        }
        for(int j=i;j<s.size();j++)//从i开始处理
            if(f[i][j]){//如果i到j为回文串
                ans.push_back(s.substr(i,j-i+1));//加入结果
                dfs(s,j+1);
                ans.pop_back();//恢复现场
            }
    }

    vector<vector<string>> partition(string s) {
        //初始化true
        f.assign(s.size(),vector<int>(s.size(),true));
        //动态规划预处理出是否为回文串
        for(int i=s.size()-1;i>=0;i--)
            for(int j=i+1;j<s.size();j++)
                f[i][j]= (s[i]==s[j]) && f[i+1][j-1];
        dfs(s,0);
        return res;
    }
};
```

### 51.N皇后

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

 

**示例 1：**

```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```
输入：n = 1
输出：[["Q"]]
```

 

**提示：**

- `1 <= n <= 9`

**思路：**按行枚举

- 确定n个位置，每次需要**回溯**和**剪枝**(即发现条件不符合时直接跳过这种情形，而不是把这种情况完成再进行排除，从而减少了时间复杂度操作。

- 条件判断包括**行、列、主对角线、副对角线**。u表示目前放置第几个皇后，也表示第几行。用列`i`作为循环条件，每次确定一行中某个位置`(u，i)`放置皇后

  - 对于行：由于每行只有一个皇后，要放置n个皇后，每次递归处理一行，放置一个皇后，当`u==n`时，结束表示放置完毕，即行为隐藏编号，无需额外的行数组

  - 对于列：设置一个列数组`col[]`，bool型，`col[i]`为true时表示当前列i已有皇后

  - 对于主对角线：同样设置一个bool数组`dg[]`，`dg[u+i]`为true时表示当前位置的主对角线方向有皇后

  - 对于副对角线：设置一个bool数组`udg[]`，`udg[i-u+n]`为true时表示当前位置的副对角线方向有皇后

    > 关于为什么是`u+i`，`i-u+n`：如果将棋盘类比成平面直角坐标系，左上角的点就是坐标原点O。可以把`u`看作横坐标，`i`看作纵坐标，若主对角线v1是不通过O的，那么v1上的点的横纵坐标之和不变，即`u+i`不变；副对角线v2上的点的横纵坐标之差不变即`|i-u|`绝对值不变，但是`i-u`会小于0（最小为0-8==-8），由于数组下标的限制，所以要对`i-u`加8。(如下图)
    > ![image-20240723155058400](D:/桌面/za/算法/assets/image-20240723155058400.png)

- 注意每次回溯前，即放置皇后后，要设置当前列、主对角、副对角为true；回溯后要设置当前列、主对角、副对角为false（即回到当前位置要恢复现场）

```c++
class Solution {
public:
    vector<vector<string>> res;
    vector<string> tmp;
    // 设置三个数组 表示对应位置是否放置皇后
    vector<bool> col;  // 列
    vector<bool> dg;   // 主对角线
    vector<bool> udg; // 副对角线

    //按行枚举 u表示行
    void dfs(int u,int n) {
        if(u==n){//处理完所有行 放满了n个皇后
            res.push_back(tmp);//存储一组结果
            return ;
        }
        //按列循环处理
        for(int i=0;i<n;i++){
        if(!col[i] && !dg[u+i] && !udg[u-i+n]){//当前位置(u,i)可以放皇后
            tmp[u][i]='Q';//放置皇后
            col[i]=dg[u+i]=udg[u-i+n]=true;//更新标志
            dfs(u+1,n);//继续递归下一行
            col[i]=dg[u+i]=udg[u-i+n]=false;//回溯后 恢复现场
            tmp[u][i]='.';
        }
    }
    }
    vector<vector<string>> solveNQueens(int n) {
        //初始化
        col.assign(n, false);  // 列
        //注意主对角线的数组大小要拉大 防止越界
        dg.assign(2*n, false);   // 主对角线
        udg.assign(2*n, false); // 副对角线
        tmp.assign(n,string(n,'.'));
        dfs(0,n);
        return res;
    }
};
```

## 二分查找

二分查找详述：[35. 搜索插入位置 - 力扣（LeetCode）](https://leetcode.cn/problems/search-insert-position/solutions/10969/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/?envType=study-plan-v2&envId=top-100-liked)

二分：[算法吧 | 快来算法吧 (suanfa8.com)](https://suanfa8.com/binary-search)

### 35.搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

 

**示例 1:**

```
输入: nums = [1,3,5,6], target = 5
输出: 2
```

**示例 2:**

```
输入: nums = [1,3,5,6], target = 2
输出: 1
```

**示例 3:**

```
输入: nums = [1,3,5,6], target = 7
输出: 4
```

 

**提示:**

- `1 <= nums.length <= 104`
- `-104 <= nums[i] <= 104`
- `nums` 为 **无重复元素** 的 **升序** 排列数组
- `-104 <= target <= 104`

**思路：**

> `(right - left) >> 1` 计算出从 `left` 到 `right` 中间位置相对于 `left` 的偏移量。也就是说，`(right - left) >> 1` 是区间长度的一半，整数部分。通过将偏移量加到 `left` 上，就得到了区间的中间位置`mid`。

> **为什么直接return left**：因为如果没有返回return middle，说明最后一定是，left>right从而跳出循环的，在此之前是left=right，如果最后是right-1导致的left>right，说明原来的right位置是大于target的，所以返回原来的right位置即left位置；如果最后是left+1导致的left>right,说明是原来的的left=right这个位置小于target，而right能移动到这个位置，说明此位置右侧是大于target的，left现在加1就移动到了这样的位置，返回left即可

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left=0,right=nums.size()-1;
        while(left<=right){
            int mid=( (right-left)>>1 )+left;//防止加法溢出int
            if(nums[mid]<target) left=mid+1;//left左边的数一定小于target
            else right=mid-1;//right右边的数一定大于等于right
        }
        //最终left=right+1
        return left;
    }
};
```

### 74.搜索二维矩阵

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target` ，如果 `target` 在矩阵中，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/mat2.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 100`
- `-104 <= matrix[i][j], target <= 104`

**思路：**

方法一：利用所给矩阵的第二条性质

从右上角开始搜索，每次淘汰一整行或一整列

右上角的数大于target，则淘汰当前列；若小于target，淘汰当前行

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = matrix[0].size();

        int i = 0, j = n - 1;

        while(i < m && j >=0) {
            if(matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else if(matrix[i][j] < target) {
                i++;
            }
        }
        return false;
    }
};
```

方法二：

若将矩阵每一行拼接在上一行的末尾，则会得到一个升序数组，我们可以在该数组上二分找到目标元素。

代码实现时，可以二分升序数组的下标，将其映射到原矩阵的行和列上。

注：此方法仅适用于每行的元素个数相等

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m=matrix.size(),n=matrix[0].size();
        int left=0,right=m*n-1;//右指针最大值为矩阵元素个数-1
        while(left<=right){
            int mid=(right-left)/2+left;
            int x=matrix[mid/m][mid%n];//注意数组下标
            if(x==target) return true;
            else if(x<target) left=mid+1;
            else right=mid-1;
        }
        return false;
    }
};
```

### 34.在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

 

**提示：**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`
- `nums` 是一个非递减数组
- `-109 <= target <= 109`

**思路**：

方法一：使用两个二分查找，找第一个等于target的位置，再找最后一个等于target的位置

第一个二分，找到等于target的值时，更新左范围first，right指针左移，继续寻找

第二个二分，找到等于target的值时，更新右范围last，left指针右移，继续寻找

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int first=-1,last=-1;//起始位置
        int left=0,right=nums.size()-1;
        //找第一个等于target的位置
        while(left<=right){
            int mid=(right-left)/2+left;
            if(nums[mid]==target){//找到一个位置
                first=mid;//赋值
                right=mid-1;//右指针左移一位 向左寻找是否还有等于target的位置
            }else if(target>nums[mid]) left=mid+1;
            else right=mid-1;
        }
        left=0,right=nums.size()-1;
        //找最后一个等于target的位置
        while(left<=right){
            int mid=(right-left)/2+left;
            if(nums[mid]==target){//找到一个位置
                last=mid;//赋值
                left=mid+1;//左指针右移一位 向右寻找是否还有等于target的位置
            }else if(target>nums[mid]) left=mid+1;
            else right=mid-1;
        }
        return {first,last};
    }
};
```

或

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int l=0,r=nums.size()-1;
        //找到第一个等于target的位置
        while(l<r){
            int mid=(l+r)/2;
            if(nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        //判断是否找到第一个位置
        if(l<0 || l>=nums.size() || nums[l]!=target) return {-1,-1};
        int start=l;//起始位置
        l=0,r=nums.size()-1;
        //找到最后一个等于target的位置
        while(l<r){
            int mid=(l+r+1)/2;
            if(nums[mid]<=target) l=mid;
            else r=mid-1;
        }
        return {start,l};
    }
};
```

方法二：复用一个二分查找`search(nums,target)`，找`>=target`的第一个

第一次二分，就直接找题目要求的`target`,`start=search(nums,target)`；

第二次二分找，找>=`target+1`的第一个，即`end=search(nums,target+1)`;

最终范围即为`(start,end-1)`

```c++
class Solution {
public:
    int search(vector<int>& nums,int target)
    {
        int l=0,r=nums.size()-1;
        while(l<=r){
            int m=(l+r)/2;
            if(nums[m]<target) l=m+1;
            else r=m-1;
        }
        return l;
    }
    vector<int> searchRange(vector<int>& nums, int target) 
    {
        int start=search(nums,target);
        if(start==nums.size()||nums[start]!=target) return {-1,-1};
        int end=search(nums,target+1)-1;
        return {start,end};
    }
};
```

### 33.搜索旋转排序数组

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```
输入：nums = [1], target = 0
输出：-1
```

 

**提示：**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- `nums` 中的每个值都 **独一无二**
- 题目数据保证 `nums` 在预先未知的某个下标上进行了旋转
- `-104 <= target <= 104`

**思路：**

找到轴点，即原数组第一个元素，旋转后分割两边为有序序列的元素下标

然后以此下标重新拼接出一个有序数组（取元素值时对数组长度取）

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int min=0,n=nums.size();
        int l=1,r=n-1;//注意l从1开始 因为min初始化为num[0]
        //找轴点
        while(l<=r){
            int mid=(l+r)/2;
            //与nums[0]比较
            if(nums[0]<nums[mid]) l=mid+1;//范围右移
            else{//若小于
                r=mid-1;//范围左移 看其左边是否还有更小的数
                min=mid;//更新当前最小值下标
            }
        }
        //找到原数组第一个元素min 分割的两边都为有序
        //将前半部分拼接到后半部分，相当于还原数组(注意每次取数要取余)
        l=min,r=l+n-1;
        //开始寻找目标值
        while(l<=r){
            int mid=(l+r)/2;
            if(target==nums[mid%n]) return mid%n;
            else if(target<nums[mid%n]) r=mid-1;
            else l=mid+1;
        }
        return -1;
    }
};
```

### 153.寻找旋转排序数组中的最小值

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 **互不相同** 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 3 次得到输入数组。
```

**示例 3：**

```
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 5000`
- `-5000 <= nums[i] <= 5000`
- `nums` 中的所有整数 **互不相同**
- `nums` 原来是一个升序排序的数组，并进行了 `1` 至 `n` 次旋转

**思路：**

和上题33思路的第一步一样，直接找轴点，用`min`记录最小值，初始化为`nums[0]`

初始化为`num[0]`是因为，若最小值`min`始终没有更新，则意味着经过了`n`次旋转，恢复了有序数组，第一个元素就为最小值

```c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int min=0;
        int l=1,r=nums.size()-1;
        while(l<=r){
            int mid=(l+r)/2;
            if(nums[0]<nums[mid]) l=mid+1;
            else{
                r=mid-1;
                min=mid;
            }
        }
        return nums[min];
    }
};
```

### 4.寻找两个正序数组的中位数

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

 

**示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

**示例 2：**

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

 

**提示：**

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-106 <= nums1[i], nums2[i] <= 106`

**思路：**[4. 寻找两个正序数组的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/median-of-two-sorted-arrays/solutions/8999/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-2/?envType=study-plan-v2&envId=top-100-liked)

假设我们要找第 `k` 小数，我们可以每次循环排除掉 `k/2` 个数。

一般的情况 A[1] ，A[2] ，A[3]，A[k/2] ... ，B[1]，B[2]，B[3]，B[k/2] ... 

若`B[k/2] < A[k/2] `

A 数组中比 `A[k/2]` 小的数有 `k/2-1 `个，B 数组中，假设 `B[k/2] `前边的数字都比 `A[k/2]` 小，也只有 `k/2-1` 个，所以比 `A[k/2] `小的数字最多有 `k/2-1+k/2-1=k-2`个，所以 `A[k/2]` 最多是第 `k-1 `小的数。而比 `A[k/2]` 小的数更不可能是第 `k` 小的数了，所以可以把它们排除。

 无论是找第奇数个还是第偶数个数字，对我们的算法并没有影响，而且在算法进行中，k 的值都有可能从奇数变为偶数，最终都会变为 1 或者由于一个数组空了，直接返回结果。

所以我们采用递归的思路，为了防止数组长度小于 `k/2`，所以每次比较 `min(k/2，len(数组))` 对应的数字，把小的那个对应的数组的数字排除，将两个新数组进入递归，并且 `k` 要减去排除的数字的个数。递归出口就是当 `k=1` 或者其中一个数字长度是 0 了。

```c++
class Solution {
public:
    //得到两个数组中第k个小的数
    int getKth(vector<int> nums1,int start1,int end1, vector<int> nums2,int start2,int end2,int k){
        //len表示各个数组每次根据条件排除后(递归)留下的元素个数
        int len1=end1-start1+1;
        int len2=end2-start2+1;
        //这里使len1始终小于len2,这样保证一点有数组为空必为len1，简化处理
        if(len1>len2) return getKth(nums2,start2,end2,nums1,start1,end1,k);//即调换顺序
        //如果一个数组为空了，那中位数就在剩下的数组nums2的start2+k-1处
        if(len1==0) return nums2[start2+k-1];
        //如果k=1，表明此时为前k-1个最小的数已经确定，现在就抉择出第k个数，即中位数
        if(k==1) return min(nums1[start1],nums2[start2]);

        //为了防止数组长度小于k/2,即防止排除k/2个元素后数组越界
        //当长度不足k/2时直接取数组的末尾元素
        int i=start1+min(len1,k/2)-1;
        int j=start2+min(len2,k/2)-1;

        //如果nums1[i]>nums2[j]，则nums2数组j索引之前的数全部排除(逻辑上)，下次自从j+1开始
        //而k则变为k - (j - start2 + 1)，即减去逻辑上排出的元素的个数(要加1，因为索引相减，相对于实际排除的时要少一个的)
        if(nums1[i]>nums2[j])
        return getKth(nums1,start1,end1,nums2,j+1,end2,k-(j-start2+1));
        else return getKth(nums1,i+1,end1,nums2,start2,end2,k-(i-start1+1));

    }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n=nums1.size(),m=nums2.size();
        //第k个数的索引k+1
        int left=(n+m+1)/2;
        int right=(n+m+2)/2;
        //若两个数组长度之和为奇数 则第k个数就是确定的一个数组内的数,left=right；否则为left和right和的1/2
        //直接两种情况一起处理 再乘1/2
        return (getKth(nums1,0,n-1,nums2,0,m-1,left)+getKth(nums1,0,n-1,nums2,0,m-1,right))*0.5;
    }
};
```

## 栈

### 20.有效的括号

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

 

**示例 1：**

**输入：**s = "()"

**输出：**true

**示例 2：**

**输入：**s = "()[]{}"

**输出：**true

**示例 3：**

**输入：**s = "(]"

**输出：**false

**示例 4：**

**输入：**s = "([])"

**输出：**true

 

**提示：**

- `1 <= s.length <= 104`
- `s` 仅由括号 `'()[]{}'` 组成

**思路：**

利用栈，遇到左括号入栈；遇到有括号，则栈顶出栈看是否与当前右括号匹配。最后栈不为空则还有左括号，返回false

```c++
class Solution {
public:
    bool isValid(string s) {
        int n=s.size();
        if(n%2!=0) return false;
        stack<char> st;
        unordered_map<char,char> pair={
            {'(',')'},
            {'{','}'},
            {'[',']'}
        };
        for(auto c:s){
            if(pair.count(c)) st.push(c);//左括号入栈
            else{//右括号 则看栈顶的左括号 是否配对
                if(st.empty() || pair[st.top()]!=c ) return false;
                st.pop();//配对的话出栈
            }
        }
        if(!st.empty()) return false;
        return true;
    }
};
```

### 155.最小栈

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类:

- `MinStack()` 初始化堆栈对象。
- `void push(int val)` 将元素val推入堆栈。
- `void pop()` 删除堆栈顶部的元素。
- `int top()` 获取堆栈顶部的元素。
- `int getMin()` 获取堆栈中的最小元素。

 

**示例 1:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

 

**提示：**

- `-231 <= val <= 231 - 1`
- `pop`、`top` 和 `getMin` 操作总是在 **非空栈** 上调用
- `push`, `pop`, `top`, and `getMin`最多被调用 `3 * 104` 次

**思路：**

方法一：使用两个栈，一个元素栈，一个辅助栈min_stack

辅助栈，与元素栈同步插入与删除，用于存储与每个元素对应的最小值。

- push() 方法： 每当push()新值进来时，如果 小于等于 min_stack 栈顶值，则一起 push() 到 min_stack，即更新了栈顶最小值；

- pop() 方法： 判断将 pop() 出去的元素值是否是 min_stack 栈顶元素值（即最小值），如果是则将 min_stack 栈顶元素一起 pop()，这样可以保证 min_stack 栈顶元素始终是 stack 中的最小值。
- getMin()方法： 返回 min_stack 栈顶即可。

```c#
class MinStack {
private:
    stack<int> st;//元素栈，正常存放各个元素
    stack<int> minSt;//辅助栈，不断存放比栈顶元素小的数
public:
    MinStack() {
    }
    
    void push(int val) {
        st.push(val);
        //辅助栈为空 或 当前值小于等于其栈顶元素
        if(minSt.empty() || minSt.top()>= val) minSt.push(val);
    }
    
    void pop() {
        int cur=st.top();
        st.pop();
        //元素栈和辅助栈栈顶元素相等时
        if(cur==minSt.top()) minSt.pop();//辅助栈才弹出栈顶元素
    }
    
    int top() {
        return st.top();
    }
    
    int getMin() {
        return minSt.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

方法二：使用一个栈[155. 最小栈 - 力扣（LeetCode）](https://leetcode.cn/problems/min-stack/solutions/42521/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-38/?envType=study-plan-v2&envId=top-100-liked)

设置一个最小值min，存储当前栈中的最小值

push()：先与min比较。小于min则先保存之前的min值（防止丢失），再更新min为当前值，再将当前值入栈；否则，直接入栈。

pop()：先得到栈顶元素并出栈。如果栈顶元素就为最小值，则min更新为出栈一次后的栈顶元素，并再出栈一次，即总共出栈两次，因为根据push操作这种情况连续存储了最小值和次小值，造成次小值是栈重复存储的。

```c++
class MinStack {
private:
    stack<int> st;
    int min=INT_MAX;//存放当前栈中的最小值
public:
    MinStack() {
    }
    
    void push(int val) {
       if(val<=min){//当前值更小
            st.push(min);//将之前的最小值入栈保存
            min=val;//更新最小值
       }
        st.push(val);//然后当前值入栈
    }
    
    void pop() {
        int cur=st.top();
        st.pop();
        if(cur==min){//如果栈顶元素就是最小值
            min=st.top();//栈顶元素重新成为最小值
            st.pop();//再弹出栈顶元素 因为上面入栈时连续存储了最小值和次小值
        }
    }
    
    int top() {
        return st.top();
    }
    
    int getMin() {
        return min;
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

### 394.字符串解码

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 `k` ，例如不会出现像 `3a` 或 `2[4]` 的输入。

 

**示例 1：**

```
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**示例 2：**

```
输入：s = "3[a2[c]]"
输出："accaccacc"
```

**示例 3：**

```
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

**示例 4：**

```
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

 

**提示：**

- `1 <= s.length <= 30`
- `s` 由小写英文字母、数字和方括号 `'[]'` 组成
- `s` 保证是一个 **有效** 的输入。
- `s` 中所有整数的取值范围为 `[1, 300]` 

**思路：**

- 构建辅助栈 stack， 遍历字符串 s 中每个字符 c；

  - **当 c 为数字时**，将数字字符转化为数字 multi，用于后续倍数计算；
  - **当 c 为字母时**，在 res 尾部添加 c；
  - **当 c 为 [ 时**，将当前 multi 和 res 入栈，并分别置空置 0：
    - 记录此` [ `前的临时结果 `res `至栈，用于发现对应 `]` 后的拼接操作；
    - 记录此 `[` 前的倍数 multi 至栈，用于发现对应 `]` 后，获取 `multi × [...] `字符串。
    - 进入到新 `[` 后，`res` 和 `multi` 重新记录。
- **当 c 为 ] 时**，stack 出栈，拼接字符串 `res = last_res + cur_multi * res`，其中:
  - `last_res`是上个 `[` 到当前 `[ `的字符串，例如 "3[a2[c]]" 中的 a；
  - `cur_multi`是当前 `[` 到 `]` 内字符串的重复倍数，例如 "3[a2[c]]" 中的 2。
- 返回字符串 res。

```c++
class Solution {
public:
    string decodeString(string s) {
        stack<pair<string,int>> st;//string存上一次拼接的字符串,int存当前[]字符串对应倍数
        string res="";//存储当前[]内字符串 也为最终结果
        int mul=0;//[]内字符串对应的倍数
        for(char c:s){
            if(c=='['){
                st.emplace(res,mul);//上一次的结果先入栈
                //然后恢复原始状态
                res="";
                mul=0;
            }else if(c==']'){//拼接之前的字符串和当前[]内的字符串 并按其倍数拼接
                string tmp=st.top().first;//上一次的字符串
                int cur_mul=st.top().second;//本次的倍数
                st.pop();
                for(int i=0;i<cur_mul;i++) tmp+=res;
                res=tmp;//更新结果
            }else if(c>='0' && c<='9'){//数字
                mul=mul*10+(c-'0');//计算倍数
            }else{//字母
                res+=c;//拼接
            }
        }
        return res;        
    }
};
```

### 739.每日温度

给定一个整数数组 `temperatures` ，表示每天的温度，返回一个数组 `answer` ，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

 

**示例 1:**

```
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
```

**示例 2:**

```
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
```

**示例 3:**

```
输入: temperatures = [30,60,90]
输出: [1,1,0]
```

 

**提示：**

- `1 <= temperatures.length <= 105`
- `30 <= temperatures[i] <= 100`

**思路：**构建**从栈底到栈顶的递减栈**

遍历整个数组，如果栈不空，且当前数字大于栈顶元素，那么如果直接入栈的话就不是 递减栈 ，所以需要取出栈顶元素，由于当前数字大于栈顶元素的数字，而且一定是第一个大于栈顶元素的数，直接求出下标差就是二者的距离。

继续看新的栈顶元素，直到当前数字小于等于栈顶元素停止，然后将数字入栈，这样就可以一直保持递减栈，且每个数字和第一个大于它的数的距离也可以算出来。

```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res(n, 0);
        stack<int> st;//栈存的是数组下标
        for (int i = 0; i < n; i++) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {//栈不为空且栈顶元素小于当前元素
                auto t = st.top();
                st.pop();
                res[t] = i - t;
            }
            st.push(i);
        }
        return res;
    }
};
```

### 84.柱状图中最大的矩形

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

**示例 1:**

![img](D:\桌面\za\算法\assets\histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**示例 2：**

![img](D:\桌面\za\算法\assets\histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

 

**提示：**

- `1 <= heights.length <=105`
- `0 <= heights[i] <= 104`

**思路：**

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        /*
        只做单调栈思路:参考"编程狂想曲"思路比较好理解
        1.核心思想:求每条柱子可以向左右延伸的长度->矩形最大宽度;矩形的高->柱子的高度
            计算以每一根柱子高度为高的矩形面积,维护面积最大值
        2.朴素的想法:遍历每一根柱子的高度然后向两边进行扩散找到最大宽度
        3.单调栈优化:因为最终的目的是寻找对应柱子height[i]右边首个严格小于height[i]的柱子height[r]
            左边同理找到首个严格小于height[i]的柱子height[l]
            维护一个单调递增栈(栈底->栈顶),那么每当遇到新加入的元素<栈顶便可以确定栈顶柱子右边界
            而栈顶柱子左边界就是栈顶柱子下面的柱子(<栈顶柱子)
            左右边界确定以后就可以进行面积计算与维护最大面积
        时间复杂度:O(N),空间复杂度:O(N)
        */
        // 引入哨兵
        // 哨兵的作用是 将最后的元素出栈计算面积 以及 将开头的元素顺利入栈
        // len为引入哨兵后的数组长度
        int len = heights.length + 2;
        int[] newHeight = new int[len];
        newHeight[0] = newHeight[len - 1] = 0;
        // [1,2,3]->[0,1,2,3,0]
        for(int i = 1; i < len - 1; i++) {
            newHeight[i] = heights[i - 1];
        }
        // 单调递增栈:存储每个柱子的索引,使得这些索引对应的柱子高度单调递增
        Stack<Integer> stack = new Stack<>();
        // 最大矩形面积
        int res = 0;
        // 遍历哨兵数组
        for(int i = 0; i < len; i++) {
            // 栈不为空且当前柱子高度<栈顶索引对应的柱子高度
            // 说明栈顶元素的右边界已经确定,就是索引为i的柱子(不含)
            // 此时将栈顶元素出栈,栈顶矩形左边界为栈顶元素下面的索引(首个小于栈顶)
            while(!stack.empty() && newHeight[i] < newHeight[stack.peek()]) {
                // 栈顶索引出栈并记录
                int pop = stack.pop();
                // 计算出栈顶元素矩形的宽度如(0,1,2)->[1,2,1],两边都不包含
                // 因此右索引-左索引-1=矩形宽度
                int w = i - stack.peek() - 1;
                // 栈顶索引对应的柱子高度就是矩形的高度
                int h = newHeight[pop];
                // 计算矩形面积
                int area = w * h;
                // 维护矩形面积最大值
                res = Math.max(res, area);
            }
            // 每当弹出一个索引就计算一个矩形面积
            // 直到当前元素>=栈顶元素(或者栈为空)时,栈顶柱子的右边界还没确定
            // 因此当前元素索引入栈即可
            stack.push(i);
        }
        return res;
    }
}
```

## 堆

### 215.数组中的第K个最大元素

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `*k*` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

 

**示例 1:**

```
输入: [3,2,1,5,6,4], k = 2
输出: 5
```

**示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

 

**提示：**

- `1 <= k <= nums.length <= 105`
- `-104 <= nums[i] <= 104`

**思路：**

方法一：建立小根堆

我们可以借助一个小顶堆来维护当前堆内元素的最小值，同时保证堆的大小为 k：

- 遍历数组将元素入堆；

- 如果当前堆内元素超过 k 了，我们就把堆顶元素去除，即去除当前的最小值。

因此我们在元素入堆的过程中，不断淘汰最小值，最终留在堆中就是数组中前 k 个最大元素，并且堆顶元素为前 k 大元素中的最小值，即为第 k 个元素。

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        //建立小根堆
        priority_queue<int,vector<int>,greater<int>> heap;
        for(auto& num:nums){
            heap.push(num);
            if(heap.size()>k){//当堆中元素数量大于k时，去掉堆顶元素
                heap.pop();
            }
        }
        //最后堆的数量为k，且堆顶即为第k大的数
        return heap.top();
    }
};
```

方法二：快速选择排序

注：快速排序的时间复杂度为O(nlogn)，而快速选择排序并不是全部排好序再选择元素，而是一边排序一边判断是否出现已经满足条件的数，平均时间复杂度为O(n)

```c++
class Solution {
public:
    //得到从小到大排序后的第k个数
    int quick_select(vector<int>& nums,int left,int right,int k){
        if(left==right) return nums[left];//数组只剩一个元素 直接返回
        int i=left-1,j=right+1;
        int x=nums[(left+right)/2];//基准元素
        while(i<j){
            while(nums[++i]<x);
            while(nums[--j]>x);
            if(i<j) swap(nums[i],nums[j]);
        }
        //左区间个数大于等于k 递归左区间
        if(k<=j) return quick_select(nums,left,j,k);
        //否则递归右区间
        else return quick_select(nums,j+1,right,k);
    }
    int findKthLargest(vector<int>& nums, int k) {
        int n=nums.size();
        return quick_select(nums,0,n-1,n-k);//第k大的数，即为递增序列第n-k个数
    }
};
```

### 347.前K个高频元素

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `k` 的取值范围是 `[1, 数组中不相同的元素的个数]`
- 题目数据保证答案唯一，换句话说，数组中前 `k` 个高频元素的集合是唯一的

**思路：**使用哈希表+大根堆

- 遍历数组，用哈希表存储，键为元素值，值为对应出现频率
- 使用大根堆存储，注意先存储频率值，再存储对应元素值，使堆先按频率值排序
- 循环取出k个堆顶元素，即为频率前k个高的元素

```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        vector<int> res;
        unordered_map<int,int> map;//键为数组元素，值为对应频率
        for(auto& num:nums){
            map[num]++;
        }
        //定义一个大根堆
        priority_queue<pair<int,int>> heap;
        //遍历哈希表 放入堆中
        for(auto it=map.begin();it!=map.end();it++){
            //注意将频率值放在第一个int,元素值放第二个 使大根堆首先按照频率值排序
            heap.push(pair<int,int>(it->second,it->first));//
        }
        while(k--){//取出频率前k高的元素
            res.push_back(heap.top().second);
            heap.pop();
        }
        return res;
    }
};
```

### 295.数据流的中位数

**中位数**是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

- 例如 `arr = [2,3,4]` 的中位数是 `3` 。
- 例如 `arr = [2,3]` 的中位数是 `(2 + 3) / 2 = 2.5` 。

实现 MedianFinder 类:

- `MedianFinder() `初始化 `MedianFinder` 对象。
- `void addNum(int num)` 将数据流中的整数 `num` 添加到数据结构中。
- `double findMedian()` 返回到目前为止所有元素的中位数。与实际答案相差 `10-5` 以内的答案将被接受。

**示例 1：**

```
输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

**提示:**

- `-105 <= num <= 105`
- 在调用 `findMedian` 之前，数据结构中至少有一个元素
- 最多 `5 * 104` 次调用 `addNum` 和 `findMedian`

**思路**：小根堆+大根堆

建立一个 小顶堆 A 和 大顶堆 B ，各保存列表的一半元素，假设元素个数为N，则规定：

- A 保存 较大 的一半，元素个数$m=\frac{N}{2}$（N为偶）或$\frac{N+1}{2}$（N为奇）
- B 保存 较小 的一半，元素个数$n=\frac{N}{2}$（N为偶）或$\frac{N-1}{2}$（N为奇）

即N为奇数时，A保存的元素个数比B多一个，即为中位数；N为偶数时，A和B的元素个数相同，中位数为两个堆顶元素的平均值。

**添加元素：**

1. 当 m=n（即 N 为 偶数）：需向 A 添加一个元素。实现方法：将新元素 num 插入至 B ，再将 B 堆顶元素插入至 A
2.  当 m≠n（即 N 为 奇数）：需向 B 添加一个元素。实现方法：将新元素 num 插入至 A ，再将 A 堆顶元素插入至 B 

> 假设插入数字 num 遇到情况 1. 由于 num 可能属于 “较小的一半” （即属于 B ），因此不能将 nums 直接插入至 A 。而应先将 num 插入至 B ，再将 B 堆顶元素插入至 A 。这样就可以始终保持 A 保存较大一半、 B 保存较小一半。
>
> 其实考虑num与B堆顶元素的比较，若大于B堆顶元素，就直接插入A；但小于B堆顶元素，就要插入B中，抉择出B堆中的最大元素，再放入A。因此直接放入B，然后取堆顶元素，就可以减少判断步骤，简化代码

```c++
class MedianFinder {
public:
    //建立小、大根堆
    priority_queue<int,vector<int>,greater<int>> A;//小根堆，存储较大的一半
    priority_queue<int> B;//大根堆，存储较小的一半
    MedianFinder() {
    }
    
    void addNum(int num) {
        if(A.size()!=B.size()){//表示当前数据个数为奇数个 加入后为偶数个
            //先将元素插入A
            A.push(num);
            //再将A的堆顶元素插入B
            B.push(A.top());
            A.pop();//弹出栈顶元素
        }else{//表示当前数据个数为偶数个 加入后为奇数个
            //先将元素插入B
            B.push(num);
            //再将B的堆顶元素插入A
            A.push(B.top());
            B.pop();//弹出
        }
    }
    
    double findMedian() {
        //数据个数为奇数 直接返回A的栈顶元素；为偶数，返回A和B栈顶元素的均值
        return A.size()!=B.size()?A.top():(A.top()+B.top())*0.5;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
```

## 贪心算法

### 121.买卖股票的最佳时机

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

 

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

 

**提示：**

- `1 <= prices.length <= 105`
- `0 <= prices[i] <= 104`

**思路**

若在前` i `天选择买入，若想达到最高利润，则一定选择价格最低的交易日买入。考虑根据此贪心思想，遍历价格列表 `prices` 并执行两步：

- 更新前` i `天的最低价格，即最低买入成本` cost`；
- 更新前` i` 天的最高利润 `profit `，即选择「前 `i−1 `天最高利润 `profit `」和「第 `i `天卖出的最高利润 `price - cost` 」中的最大值 ；

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0,cost=INT_MAX;//利润和最小成本
        for(auto price:prices){
            cost=min(cost,price);
            res=max(res,price-cost);
        }
        return res;
    }
};
```

### 55.跳跃游戏

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置**可以跳跃的最大长度**。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

 

**提示：**

- `1 <= nums.length <= 104`
- `0 <= nums[i] <= 105`

**思路**：动态规划

如果某一个作为 **起跳点** 的格子可以跳跃的距离是 3，那么表示后面 3 个格子都可以作为 **起跳点**

- `k[i]`表示当前下标`i`能达到的最远距离（下标）。（实际上的最远距离上限为`num.size()-1`，即从第一个下标到最后一个下标的距离）
- 遍历`i`，对于上一次位置所能到达的最远距离`k[i-1]`，若小于`i`，意味着从上一次的位置无论如何都到达不了当前下标为`i`的位置，更遑论终点，直接返回`false`；若大于等于`i`，意味着上一步的位置有机会到达`i`，更新一下当前可能达到的最远距离（下标）`k[i]=max(k[i-1],i+num[i])`
- 遍历完就返回`true`，能到达最后一个下标

对于`k[i]`可空间优化为一个变量`k`

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int k=0;//表示当前所能到达的最远距离 
        for(int i=0;i<nums.size();i++){
            //遍历中出现下标i大于当前所能达到的最远距离 即不可达 直接返回false
            if(i>k) return false;
            //否则还有希望
            k=max(k,i+nums[i]);//k更新为当前所能达到的最远距离
        }
        //遍历完k始终大于等于i 即可达到末尾
        return true;
    }
};
```

### 45.跳跃游戏II

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

 

**示例 1:**

```
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**示例 2:**

```
输入: nums = [2,3,0,1,4]
输出: 2
```

 

**提示:**

- `1 <= nums.length <= 104`
- `0 <= nums[i] <= 1000`
- 题目保证可以到达 `nums[n-1]`

**思路：**

- 如果某一个作为 **起跳点** 的格子可以跳跃的距离是 3，那么表示后面 3 个格子都可以作为 **起跳点**。
- 可以对每一个能作为 **起跳点** 的格子都尝试跳一次，把 **能跳到最远的距离** 不断更新。
- 如果从这个 **起跳点** 起跳叫做第 1 次 跳跃，那么从后面 3 个格子起跳 都 可以叫做第 2 次 跳跃。
- 所以，当一次 **跳跃** 结束时，从下一个格子开始，到现在 **能跳到最远的距离**，都 是下一次 跳跃 的 **起跳点**。

1. 对每一次 **跳跃** 用 while 循环来模拟。
2. 跳完一次之后，更新下一次 起跳点 的范围。
3. 在新的范围内跳，更新 能跳到最远的距离。
4. 记录 跳跃 次数，如果跳到了终点，就得到了结果。

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int res=0;//结果
        int start=0,end=1;//表示当前可尝试跳跃点的区间范围(start,end-1) 初始就只有下标0
        while(end<nums.size()){
            //表示当前可尝试跳跃的点中所能达到的最远距离
            int maxPos=0;
            //遍历这些点
            for(int i=start;i<end;i++){
                maxPos=max(maxPos,i+nums[i]);
            }
            //更新下一次可跳跃点的区间范围
            start=end;//指向下一次的第一个跳跃点
            end=maxPos+1;//end始终指向最后一个跳跃点的后一个位置
            res++;//跳数+1
        }
        return res;
    }
};
```

**进一步优化**

从上面代码观察发现，其实被 while 包含的 for 循环中，`i `是从头跑到尾的（即每次更新的区间都是不重叠的，`i`就从`0`遍历到`n-2`）

1. 只需要在一次 **跳跃** 完成时，更新下一次 **能跳到最远的距离**。
2. 并以此刻作为时机来更新 **跳跃** 次数。
3. 就可以在一次 for 循环中处理。

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int res=0;//结果
        int end=0;//边界范围，即当前所能到达的最远下标
        int maxPos=0;//当前边界范围内所有点所能跳跃的最远距离
        for(int i=0;i<nums.size()-1;i++){//i无需遍历到最后一个n-1,到那时就不用跳了
            //不断循环 完成一组跳跃点 更新最远距离
            maxPos=max(maxPos,i+nums[i]);
            if(i==end){//i走到这一组的边界范围 则更新下一组的范围
                end=maxPos;
                res++;//跳数+1
            }
        }
        return res;
    }
};
```

### 763.划分字母区间

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。

 

**示例 1：**

```
输入：s = "ababcbaca defegde hijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 
```

**示例 2：**

```
输入：s = "eccbbbbdec"
输出：[10]
```

 

**提示：**

- `1 <= s.length <= 500`
- `s` 仅由小写英文字母组成

**思路：**

由于同一个字母只能出现在同一个片段，显然同一个字母的第一次出现的下标位置和最后一次出现的下标位置必须出现在同一个片段。因此需要遍历字符串，得到每个字母最后一次出现的下标位置。

在得到每个字母最后一次出现的下标位置之后，可以使用贪心的方法将字符串划分为尽可能多的片段，具体做法如下。

- 从左到右遍历字符串，遍历的同时维护当前片段的开始下标 `start `和结束下标 `end`，初始时 `start=end=0`。

- 对于每个访问到的字母 c，得到当前字母的最后一次出现的下标位置`end_c` ，则当前片段的结束下标一定不会小于 `end_c `，因此令 `end=max(end,end_c)`。
- 当访问到下标 `end` 时，当前片段访问结束，当前片段的下标范围是` [start,end]`，长度为 `end−start+1`，将当前片段的长度添加到返回值，然后令 `start=end+1`，继续寻找下一个片段。

- 重复上述过程，直到遍历完字符串。


```c++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        vector<int> last(26);//记录每个字母最后一次出现的位置
        int n=s.length();
        for(int i=0;i<n;i++){
            last[s[i]-'a']=i;
        }
        vector<int> res;
        int start=0,end=0;//所分割段的起点与终点
        for(int i=0;i<n;i++){
            end=max(end,last[s[i]-'a']);//得到当前段中所有字母最大的最后一个位置
            if(i==end){//即当前可分割 当前段中的字母在后面均不会再出现
                res.push_back(end-start+1);
                start=end+1;
            }
        }
        return res;
    }
};
```

## 动态规划

### 70.爬楼梯

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

 

**示例 1：**

```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

**示例 2：**

```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

 

**提示：**

- `1 <= n <= 45`

**思路：**动态规划

设跳上 `n` 级台阶有 `f(n) `种跳法。在所有跳法中，青蛙的最后一步只有两种情况： 跳上 `1`级或 `2`级台阶。

- 当为 `1`级台阶： 剩 `n−1 `个台阶，此情况共有` f(n−1) `种跳法。

- 当为` 2` 级台阶： 剩` n−2` 个台阶，此情况共有` f(n−2) `种跳法。

即` f(n) `为以上两种情况之和，即 `f(n)=f(n−1)+f(n−2) `，即状态转移方程

`f(0)=1 `, `f(1)=1` , `f(2)=2` 

由于对于第`n`项仅与前面两项`n-1`,`n-2`有关，所以只需要初始化三个整形变量 `sum`, `a`, `b` ，利用辅助变量 *s**u**m* 使 *a*,*b* 两数字交替前进即可，是空间复杂度由$O(n)$降为$O(1)$

```c++
class Solution {
public:

    int climbStairs(int n) {
        int a=1,b=1;//表示第0级和第1级台阶的爬法
        int sum=0;//辅助变量
        for(int i=2;i<=n;i++){
            sum=a+b;//f[n]=f[n-1]+f[n-2]
            a=b;//后移
            b=sum;//后移
        }
        return b;
    }
};
```

### 118.杨辉三角

给定一个非负整数 *`numRows`，*生成「杨辉三角」的前 *`numRows`* 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

![img](D:\桌面\za\算法\assets\1626927345-DZmfxB-PascalTriangleAnimated2.gif)

 

**示例 1:**

```
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

**示例 2:**

```
输入: numRows = 1
输出: [[1]]
```

 

**提示:**

- `1 <= numRows <= 30`

**思路：**动态规划

- 设置状态数组`f[i][j]`，表示当前元素`(i,j)`的值，`i`为行，`j`为列。（类似下图的编号，行为红色，列为绿色）

  <img src="assets\86c68268b08ecc49a7dfa494c0fc2c6f.png" alt="img" style="zoom:50%;" />

- 右杨辉三角形的性质：每一行的第一个元素和最后一个元素都为`1`

- 按每行处理，遍历到每行就扩充当前行的数组大小，由于第一个元素和最后一个元素已固定，只需计算下标从`1`到`i-1`的元素.

- 状态转移方程为`f[i][j]=f[i-1][j-1]+f[i-1][j]`

```c++
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res(numRows);
        for(int i=0;i<numRows;i++){
            res[i].resize(i+1);//设置每一行的数组大小
            res[i][0]=res[i][i]=1;//每行的第一个元素和最后一个元素总是1
            //计算每一行
            for(int j=1;j<i;j++){
                //上一行前一列的数+上一行当前列的数
                res[i][j]=res[i-1][j-1]+res[i-1][j];
            }
        }
        return res;
    }
};
```

### 198.打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

 

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

 

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 400`

**思路**

首先考虑最简单的情况。如果只有一间房屋，则偷窃该房屋，可以偷窃到最高总金额。如果只有两间房屋，则由于两间房屋相邻，不能同时偷窃，只能偷窃其中的一间房屋，因此选择其中金额较高的房屋进行偷窃，可以偷窃到最高总金额。

如果房屋数量大于两间，对于第 k (k>2) 间房屋，有两个选项：

- 偷窃第 k 间房屋，那么就不能偷窃第 k−1 间房屋，偷窃总金额为前 k−2 间房屋的最高总金额与第 k 间房屋的金额之和。

- 不偷窃第 k 间房屋，偷窃总金额为前 k−1 间房屋的最高总金额。


在两个选项中选择偷窃总金额较大的选项，该选项对应的偷窃总金额即为前 k 间房屋能偷窃到的最高总金额。

用 `dp[i] `表示前` i `间房屋能偷窃到的最高总金额，那么就有如下的状态转移方程：

`dp[i]=max(dp[i−2]+nums[i],dp[i−1])`

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n=nums.size();
        //dp[i]表示偷i间房能偷到的最大金额
        vector<int> dp(n+1,0);
        dp[0]=0;
        dp[1]=nums[0];
        for(int i=2;i<=n;i++){
            //max(偷前i-1间房，偷前i-2间房+最后一间房)
            dp[i]=max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[n];
    }
};
```

### 279.完全平方数

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

 

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

 

**提示：**

- `1 <= n <= 104`

**思路：**动态规划

- 首先初始化长度为` n+1` 的数组 `dp`，每个位置都为` 0`

- 对数组进行遍历，下标为 `i`，每次都将当前数字先更新为最大的结果，即 `dp[i]=i`，比如 `i=4`，最坏结果为` 4=1+1+1+1` 即为 `4 `个数字，即4个完全平方数的和
- 动态转移方程为：`dp[i] = MIN(dp[i], dp[i - j * j] + 1)`，`i `表示当前数字，`j*j `表示平方数

```c++
class Solution {
public:
    int numSquares(int n) {
        //dp[i]表示和为i的当前最少完全平方数
        vector<int> dp(n+1,0);
        for(int i=1;i<=n;i++){
            dp[i]=i;//初始化为最坏的结果
            for(int j=1;i-j*j>=0;j++){//然后对于和为i寻找最少完全平方数
                dp[i]=min(dp[i],dp[i-j*j]+1);//动态转移方程
            }
        }
        return dp[n];
    }
};
```

### 322.零钱兑换

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

 

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins = [1], amount = 0
输出：0
```

 

**提示：**

- `1 <= coins.length <= 12`
- `1 <= coins[i] <= 231 - 1`
- `0 <= amount <= 104`

**思路：**动态规划

定义 `F(i)` 为组成金额` i `所需最少的硬币数量

$F(i)$ 对应的转移方程应为$F(i)=min_{j=0…n−1}F(i−c_j)+1$其中 $c_j $代表的是第` j` 枚硬币的面值，即我们枚举最后一枚硬币面是$c_j$，那么需要从$ i−c_j$这个金额的状态$ F(i−c_j) $转移过来，再算上枚举的这枚硬币数量` 1` 的贡献，由于要硬币数量最少，所以 $F(i) $为前面能转移过来的状态的最小值加上枚举的硬币数量` 1 `。

```c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        //dp[i]表示和为i的最少硬币，初始化最少硬币数为amount+1
        vector<int> dp(amount+1,amount+1);
        dp[0]=0;//金额为0，需要硬币数为0
        for(int i=1;i<=amount;i++){
            //对于每一步i，遍历计算是从哪个硬币转移过来的硬币数最少
            for(int j=0;j<coins.size();j++){
                if(i>=coins[j])//保证i大于coins[j]才进行转移
                    dp[i]=min(dp[i],dp[i-coins[j]]+1);
            }
        }
        //如果最终的结果大于金额 则意味着凑不到该金额 返回-1
        return dp[amount]>amount?-1:dp[amount];
    }
};
```



### 139.单词拆分

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。

**注意：**不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

 

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

 

**提示：**

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` 和 `wordDict[i]` 仅由小写英文字母组成
- `wordDict` 中的所有字符串 **互不相同**

**思路：**动态规划

1. 初始化 `dp=[False,⋯,False]`，长度为 `n+1`。`n `为字符串长度。`dp[i] `表示 `s `的前 `i `位是否可以用 `wordDict `中的单词表示。`dp`首字符为空字符，从第二个位置才开始表示`s`字符串

2. 初始化 `dp[0]=True`，空字符可以被表示。
3. 遍历字符串的所有子串，遍历开始索引 `i`，遍历区间` [0,n)`：
   - 遍历结束索引 `j`，遍历区间` [i+1,n+1)`：
     - 若` dp[i]=True` 且 `s[i,⋯,j)` 在 `wordDict` 中：`dp[j]=True`。解释： `s[i,⋯,j)` 出现在 `wordDict `中，说明 `s` 的从`i`到 `j `位可以表示。

4. 返回 `dp[n]`

```c#
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        //dp[i]为true表示前i位可以被wordDict中的字符串表示
        vector<bool> dp(s.size()+1,false);
        dp[0]=true;//首字符为空字符 可以被表示 从1开始存储s
        for(int i=0;i<s.size();i++){
            if(!dp[i]) continue;//不可被表示 继续向后一个字符查看
            //碰到可以被表示时
            for(auto& word:wordDict){
                //s[i~i+word.size()]可被s中的word表示
                if(word.size()+i<=s.size() && s.substr(i,word.size())== word) dp[i+word.size()]=true;
            }
        }
        return dp[s.size()];
    }
};
```

### 300.最长递增子序列

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**子序列** 是可以通过从另一个数组删除或不删除某些元素，但不更改其余元素的顺序得到的数组。

 

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**示例 2：**

```
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

 

**提示：**

- `1 <= nums.length <= 2500`
- `-104 <= nums[i] <= 104`

**思路：**

- 一维数组`dp[i]`表示以第`i`个数为结尾的最长递增子序列的长度。
- 状态划分：选定`i`为结尾的递增子序列，则再从`[0,i-1]`中筛选出倒数第二个位置的数，使递增子序列的长度最大。注意这个倒数第二个位置的**数必须满足`nums[j]<nums[i]`，这样才能保证递增序列**。
- 状态转移方程为`dp[i]=max(dp[i],dp[j]+1);`

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        //dp[i]表示以i为结尾的数的最大递增子序列长度
        vector<int> dp(n,1);//初始化为1，即仅包含自身
        for(int i=1;i<n;i++){
            for(int j=0;j<i;j++)
                if(nums[j]<nums[i]) dp[i]=max(dp[i],dp[j]+1);
        }
        int res=0;
        for(int i=0;i<n;i++)   
            res=max(res,dp[i]);
        return res;
    }
};
```

**二分优化：**

- 首先在上述解法的基础上，假如存在一个序列3 1 2 5，以3结尾的上升子序列长度为1，以1为结尾的上升子序列长度也为1，这是**两个长度一样的上升子序列**（伏笔：**结尾元素1<3**）。在继续向后遍历查找时，看3这个序列，当出现一个比3大的数时，以3结尾的上升子序列就会更新，比如遍历到5了，那么上升序列变为3 5；同时注意到这个5一定会加入到以1结尾的上升序列中（因为1<3，那么1<5的），那么含有1的上升序列长度一定是>=2的，因为中间可能存在<3但>1的数（比如这里就有2，序列长度就更新为3）。可以看出存在3的这个序列就不需要枚举了，**因为存在1的序列往后遍历的长度是一定大于你这个存在3的序列的（前提是以1结尾和以3结尾的上升序列长度相等），那我找最长的时候怎么都轮不到包含3的序列头上**，那我一开始在1和3结尾的序列之后直接舍弃枚举包含3的序列了（去掉冗余）。
- 在以上的分析得到：**当存在两个上升序列长度相同时，结尾数更大的序列可以舍去不再枚举，所以每次就干脆选出相同长度结尾元素最小的序列继续操作**
- 那么**状态表示**更改为：`f[i]`**表示长度为`i+1`(因为下标从0开始)的最长上升子序列，末尾最小的数字**。(**所有长度为`i+1`**的最长上升子序列所有结尾中，结尾最小的数) 即长度为`i`的子序列末尾最小元素是什么。
- **状态计算**：对于每一个数`w[i]`, **如果大于`f[cnt-1]`**(下标从`0`开始，`cnt`长度的最长上升子序列，末尾最小的数字)，那就将这个数`w[i]`添加到当前序列末尾，使得最长上升序列长度`+1`(`cnt++`)，当前末尾最小元素变为`w[i]`。 **若`w[i]`小于等于`f[cnt-1]`,**说明不会更新当前序列的长度，**但之前某个序列末尾的最小元素要发生变化**，找到第一个 **大于或等于(不能直接写大于，要保证单增)** `w[i]`的数的位置`k`，将这个数`w[i]`放在`k`的位置（**其实就是找到`w[i]`适合存在的位置，这里就使用二分查找，更新保证长度为k+1的序列的末尾元素为最小，即`f[k]=w[i]`**）。
- 查找位置的过程使用**二分查找**

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int res=0,n=nums.size();
        //dp[i]表示所有长度为i+1的子序列中末尾元素的最小值
        vector<int> dp(n);
        dp[res++]=nums[0];//dp[0]=nums[0]
        for(int i=0;i<n;i++){
            //如果当前值大于该长度递增序列末尾最小元素，则加入该长度的子序列中，res++
            if(nums[i]>dp[res-1]) dp[res++]=nums[i];
            else{//否则踢掉末尾元素 寻找合适的位置插入num[i]
                //二分查找合适位置
                int left=0,right=res-1;
                while(left<right){
                    int mid=(right-left)/2+left;
                    if(nums[i]<=dp[mid]) right=mid;
                    else left=mid+1;
                }
                //找到合适位置了 插入
                dp[left]=nums[i];
            }
        }
        return res;
    }
};
```

### 152.乘积最大子数组

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

**子数组** 是数组中连续的 **非空** 元素序列

测试用例的答案是一个 **32-位** 整数。

 

**示例 1:**

```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

 

**提示:**

- `1 <= nums.length <= 2 * 104`
- `-10 <= nums[i] <= 10`
- `nums` 的任何子数组的乘积都 **保证** 是一个 **32-位** 整数

**思路：**

- 遍历数组时计算当前最大值，不断更新

- 令`imax`为当前最大值，则当前最大值为` imax = max(imax * nums[i], nums[i])`
- 由于存在负数，那么会导致最大的变最小的，最小的变最大的。因此还需要维护当前最小值`imin`，`imin = min(imin * nums[i], nums[i])`
- 当负数出现时则`imax`与`imin`进行交换再进行下一步计算

```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int res=INT_MIN;//存储结果
        int imax=1,imin=1;//现阶段的最大值和最小值
        for(int num:nums){
            //如果当前元素为负数，则最大值会变最小值，最小值会变最大值
            if(num<0){//那么交换一下现阶段的最大值和最小值
                int tmp=imax;
                imax=imin;
                imin=tmp;
            }
            //和当前元素比较 更新现阶段最大最小值
            //为什么和当前元素比较：从前往后一直乘过来的，也可能出现前面的乘积乘以当前元素反倒小于当前元素，如前面乘积都是0的情况
            imax=max(imax*num,num);
            imin=min(imin*num,num);
            
            //更新全体最大值
            res=max(res,imax);
        }
        return res;
    }
};
```

### 416.分割等和子集

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

 

**示例 1：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

**示例 2：**

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

 

**提示：**

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`

**思路：**动态规划，类似于0-1背包问题

在使用动态规划求解之前，首先需要进行以下判断。

- 根据数组的长度 `n` 判断数组是否可以被划分。如果 `n<2`，则不可能将数组分割成元素和相等的两个子集，因此直接返回 `false`。

- 计算整个数组的元素和 `sum `以及最大元素 `maxNum`。如果 `sum` 是奇数，则不可能将数组分割成元素和相等的两个子集，因此直接返回 `false`。如果 `sum `是偶数，则令` target= sum/2` ，需要判断是否可以从数组中选出一些数字，使得这些数字的和等于 `target`。如果` maxNum>target`，则除了 `maxNum `以外的所有元素之和一定小于 `target`，因此不可能将数组分割成元素和相等的两个子集，直接返回` false`。

创建二维数组 `dp`，包含 `n `行 `target+1` 列，其中 `dp[i][j] `表示从数组的 `[0,i] `下标范围内选取若干个正整数（可以是 0 个），是否存在一种选取方案使得被选取的正整数的和等于` j`。初始时，`dp `中的全部元素都是 `false`。

在定义状态之后，需要考虑边界情况。以下两种情况都属于边界情况。

- 如果不选取任何正整数，则被选取的正整数之和等于` 0`。因此对于所有 `0≤i<n`，都有 `dp[i][0]=true`。

- 当` i==0 `时，只有一个正整数 `nums[0]` 可以被选取，因此 `dp[0][nums[0]]=true`。


对于` i>0` 且 `j>0 `的情况，如何确定 `dp[i][j] `的值？需要分别考虑以下两种情况。

- 如果` j≥nums[i]`，则对于当前的数字 `nums[i]`，可以选取也可以不选取

  - 如果不选取` nums[i]`，则 `dp[i][j]=dp[i−1][j]`；
  - 如果选取 `nums[i]`，则 `dp[i][j]=dp[i−1][j−nums[i]]`

  两种情况只要有一个为 `true`，就有 `dp[i][j]=true`。

- 如果` j<nums[i]`，则在选取的数字的和等于 `j` 的情况下无法选取当前的数字 `nums[i]`，因此有 `dp[i][j]=dp[i−1][j]`。

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n=nums.size();
        if(n<2) return false;//元素个数小于2 不可能分割出两个相等的子数组
        int sum=0,maxNum=INT_MIN;
        for(auto num:nums){
            sum+=num;//求数组所有元素之和
            maxNum=max(maxNum,num);//得到数组的最大值
        } 
        //通过位运算来判断数组之和为奇数还是偶数 比单纯的逻辑运算sum%2!=0 快
        if(sum & 1) return false;//若为奇数，则返回false

        int target=sum/2;
        //若数组之和的一般小于最大值 则依旧不可分割出两个相等子数组
        if(maxNum>target) return false;

        //开始动态规划
        //dp[i][j]:[0,i]内是否存在取若干个(可为0个)元素和为j的方案，初始为0即false
        vector<vector<int>> dp(n,vector<int>(target+1,0));

        //对于任意[0,i]一定能取到和为0的方案（即不取元素）
        for(int i=0;i<n;i++) dp[i][0]=true;
        //i=0时，仅能取到和为nums[0]的方案
        dp[0][nums[0]]=true;

        for(int i=1;i<n;i++)
            //得到各种和的方案是否存在 直到和为target，即sum/2
            for(int j=1;j<=target;j++){
                if(j>=nums[i])//只要有其中一种情况存在就能得到和为j的方案
                    dp[i][j]=dp[i-1][j] | dp[i-1][j-nums[i]];//将当前元素加入或不加入
                else//和小于nums[i] 自然不能将nums[i]加入当前子数组
                    dp[i][j]=dp[i-1][j];
            }
        return dp[n-1][target];//能凑出和为数组总和一半的方案
    }
};
```

空间优化

可以发现在计算 `dp` 的过程中，每一行的 `dp `值都只与上一行的` dp` 值有关，因此只需要一个一维数组即可将空间复杂度降到 `O(target)`。此时的转移方程为：`dp[j]=dp[j] ∣ dp[j−nums[i]]`

且需要注意的是第二层的循环我们需要从大到小计算（即逆序遍历，类似0-1背包问题），因为如果我们从小到大更新 `dp` 值，那么在计算 `dp[j]` 值的时候，`dp[j−nums[i]]` 已经是被更新过的状态，不再是上一行的` dp `值。

```c#
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n=nums.size();
        if(n<2) return false;//元素个数小于2 不可能分割出两个相等的子数组
        int sum=0,maxNum=INT_MIN;
        for(auto num:nums){
            sum+=num;//求数组所有元素之和
            maxNum=max(maxNum,num);//得到数组的最大值
        } 
        //通过位运算来判断数组之和为奇数还是偶数 比单纯的逻辑运算sum%2!=0 快
        if(sum & 1) return false;//若为奇数，则返回false

        int target=sum/2;
        //若数组之和的一般小于最大值 则依旧不可分割出两个相等子数组
        if(maxNum>target) return false;

        //开始动态规划
        //优化为一维数组dp[j]:是否存在取若干个(可为0个)元素和为j的方案
        vector<int> dp(target+1,0);

        //一定能取到和为0的方案（即不取元素）
        dp[0]=true;
        for(int i=1;i<n;i++)
            //逆序遍历 从最大值target开始 防止dp[j-nums[i]]在dp[j]前被更新过（喜欢原本的 没改变的）
            for(int j=target;j>=nums[i];j--){//最小值为nums[i] 省去判断j和当前元素的大小
                    dp[j] |=dp[j-nums[i]];
            }
        return dp[target];//能凑出和为数组总和一半的方案
    }
};
```

### 32.最长有效括号

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串（**子字符串** 是字符串中连续的字符序列）的长度。

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

 

**提示：**

- `0 <= s.length <= 3 * 104`
- `s[i]` 为 `'('` 或 `')'`

**思路：**

方法一：动态规划

我们定义` dp[i] `表示以下标 `i` 字符结尾的最长有效括号的长度。我们将` dp` 数组全部初始化为 `0` 。显然有效的子串一定以` ‘)’ `结尾，因此我们可以知道以` ‘(’ `结尾的子串对应的` dp `值必定为 `0` ，我们只需要求解 `‘)’` 在 `dp `数组中对应位置的值。

我们从前往后遍历字符串求解` dp `值，我们每两个字符检查一次：

1. `s[i]=‘)’ `且 `s[i−1]=‘(’`，也就是字符串形如` “……()”`，我们可以推出：`dp[i]=dp[i−2]+2`

   我们可以进行这样的转移，是因为结束部分的 "()" 是一个有效子字符串，并且将之前有效子字符串的长度增加了 2 。注意`i>=2`，防止数组越界

2. `s[i]=‘)’ `且` s[i−1]=‘)’`，也就是字符串形如` “……))”`，我们可以推出：

   如果 `s[i−dp[i−1]−1]=‘(’`，那么`dp[i]=dp[i−1]+dp[i−dp[i−1]−2]+2`

   > 这里我们考虑如果倒数第二个 `‘)’ `是一个有效子字符串的一部分（记作 `subs`），对于最后一个` ‘)’` ，如果它是一个更长子字符串的一部分，那么它一定有一个对应的` ‘(’ `，且它的位置在倒数第二个` ‘)’ `所在的有效子字符串的前面（也就是 `subs`的前面）。因此，如果子字符串 `subs`的前面恰好是` ‘(’ `，那么我们就用 2 加上 `subs` 的长度（即`dp[i−1]`）去更新 `dp[i]`。同时，我们也会把有效子串` “(sub s)”` 之前的有效子串的长度也加上，也就是再加上 `dp[i−dp[i−1]−2]`。总而言之就是判断是否出现了`"((.......))"`的情况

最后的答案即为` dp `数组中的最大值。

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int res=0,n=s.size();
        //dp[i]：以i为结尾的最长有效括号的长度
        vector<int> dp(n,0);
        for(int i=1;i<n;i++){
            if(s[i]==')'){
                if(s[i-1]=='(')//刚好凑出一对'()'
                    dp[i]=(i>=2?dp[i-2]:0)+2;//注意判断i>=2 防止数组越界
                else if(i-dp[i-1]>0 && s[i-dp[i-1]-1]=='(')
                    dp[i]=dp[i-1]+((i-dp[i-1])>=2?dp[i-dp[i-1]-2]:0)+2;
                    res=max(res,dp[i]);//更新最大长度
            }
        }
        return res;
    }
};
```

方法二：栈

具体做法是我们始终保持**栈底元素**为当前已经遍历过的元素中「**最后一个没有被匹配的右括号的下标**」，这样的做法主要是考虑了边界条件的处理，栈里其他元素维护左括号的下标：

- 对于遇到的每个` ‘(’ `，我们将它的下标放入栈中
- 对于遇到的每个` ‘)’ `：
  - 如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
  - 如果栈不为空且栈顶元素为左括号，则弹出栈顶元素与当前括号匹配
    - 若弹出栈顶元素后，栈不为空，则当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
    - 若弹出栈顶元素后，栈为空，说明从头开始的序列都是有效的，那么长度就是当前下标+1

我们从前往后遍历字符串并更新答案即可。

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int res=0;
        //栈 存储左括号'('的数组下标
        stack<int> st;
        for(int i=0;i<s.size();i++){
            if(s[i]=='(') st.push(i);//是左括号的话入栈
            else{//右括号
                if(!st.empty() && s[st.top()]=='('){//栈顶与当前右括号匹配
                    st.pop();//匹配完出栈
                    //这里若栈为空了，说明从头开始的序列都是有效的，那么长度就是当前序号+1
                    if(st.empty()) res=max(res,i+1);
                    //否则长度i-栈顶元素
                    else res=max(res,i-st.top());
                }else{//此时栈为空 或 为右括号
                    st.push(i);//当前右括号入栈
                }
            }
        }
        return res;
    }
};
```

## 多维动态规划

### 62.不同路径

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

 

**示例 1：**

![img](D:\桌面\za\算法\assets\1697422740-adxmsI-image.png)

```
输入：m = 3, n = 7
输出：28
```

**示例 2：**

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

**示例 3：**

```
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```
输入：m = 3, n = 3
输出：6
```

 

**提示：**

- `1 <= m, n <= 100`
- 题目数据保证答案小于等于 `2 * 109`

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        //dp[i][j]:到达（i,j）的路径数
        vector<vector<int>> dp(m,vector<int>(n,0));
        //对于上边界和左边界 只能有一条路径 因为机器人只能下移或者右移
        for(int i=0;i<n;i++) dp[0][i]=1;
        for(int i=0;i<m;i++) dp[i][0]=1;
        //动态规划 从第二行第二列开始
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                //只能从上方或右方到达当前位置
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
        return dp[m-1][n-1];
    }
};
```

空间优化成一维：当前位置的状态仅和当前列`j`和上一列`j-1`相关

`dp[j]=dp[j]+dp[j-1]`，`dp[j]`是从上方过来，`dp[j-1]`是从左方过来

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n,1);//初始化为1
        //动态规划 从第二行第二列开始
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                //只能从上方或右方到达当前位置
                dp[j]=dp[j]+dp[j-1];
        return dp[n-1];
    }
};
```

### 64.最小路径和

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg)

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

 

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 200`
- `0 <= grid[i][j] <= 200`

**思路：**

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m=grid.size(),n=grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        dp[0][0]=grid[0][0];
        for(int i=1;i<m;i++) dp[i][0]=dp[i-1][0]+grid[i][0];
        for(int i=1;i<n;i++) dp[0][i]=dp[0][i-1]+grid[0][i];
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j];
        return dp[m-1][n-1];
    }
};
```

空间优化为一维

[64. 最小路径和 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-path-sum/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m=grid.size(),n=grid[0].size();
        //向矩阵的上方和右方补上一行一列 避免边界条件的处理
        //则实际位置是从dp[1][1]出发
        //这里直接优化为一维 初始化为最大值
        vector<int> dp(n+1,INT_MAX);
        dp[1]=0;//保证到达grid[0][0]的最小路径和为它本身
        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++)
                //因为补上的一行一列 所以+grid[i-1][j-1]
                dp[j]=min(dp[j],dp[j-1])+grid[i-1][j-1];
        return dp[n];
    }
};
```

### 5.最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的 回文子串(如果字符串向前和向后读都相同，则它满足 **回文性**)。



**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"
```

 

**提示：**

- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成

**思路：**

对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。例如对于字符串 “ababa”，如果我们已经知道 “bab” 是回文串，那么 “ababa” 一定是回文串，这是因为它的首尾两个字母都是 “a”。

- 根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 `P(i,j) `表示字符串 `s` 的第` i `到` j `个字母组成的串（下文表示成 `s[i:j]`）是否为回文串：` P(i,j) =true`即`s[i...j]`为回文串
- 由此得到动态规划方程：`P(i,j)=P(i+1,j-1)^(s[i]==s[j])`

也就是说，只有 `s[i+1:j−1]` 是回文串，并且` s` 的第 `i `和 `j `个字母相同时，`s[i:j]` 才会是回文串。

我们还需要考虑动态规划中的边界条件，即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，只要它的两个字母相同，它就是一个回文串。

根据这个思路，我们就可以完成动态规划了，最终的答案即为所有` P(i,j)=true` 中` j−i+1`（即子串长度）的最大值。注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。

```c#
class Solution {
public:
    string longestPalindrome(string s) {
        int n=s.size();
        if(n<2) return s;
        //分别记录长度和回文串的起始下标
        int res=1,begin=0;
        //dp[i][j]:表示s[i..j]为回文串
        vector<vector<int>> dp(n,vector<int>(n,0));
        for(int i=0;i<n;i++) dp[i][i]=true;//长度为1的串都是回文串

        //递推开始
        for(int L=2;L<=n;L++)//枚举子串长度2~n
            for(int i=0;i<n;i++){//子串左边界
                //得到右边界 由j-i+1=L得
                int j=L+i-1;
                //如果越过右边界就退出当前循环 枚举下一个长度的子串
                if(j>=n) break;

                //判断是否是回文串
                if(s[i]!=s[j]) 
                    dp[i][j]=false;//子串首尾不等
                else//首尾相等
                    //如果当前区间长度<4，即长度为2或3，除开首尾只有0个或1个字符 包是回文串；否则看除开首尾中间的子串是否是回文串
                    dp[i][j]=l<4?true:dp[i+1][j-1];
                
                //dp[i][j]=true为回文串 且 长度大于之前的最大长度 则更新
                if(dp[i][j] && j-i+1>res){
                    res=j-i+1;
                    begin=i;//更新起始下标
                }
            }
        return s.substr(begin,res);
    }
};
```

### 1143.最长公共子序列

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

 

**示例 1：**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

**示例 2：**

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
```

**示例 3：**

```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。
```

 

**提示：**

- `1 <= text1.length, text2.length <= 1000`
- `text1` 和 `text2` 仅由小写英文字符组成。

**思路：**

假设字符串 text1、和 text2的长度分别为`m`和`n`，创建`m＋1`行`n＋1`列的二维数组`dp`，其中 `dp[i][j]`表示 text1前`i`个字符和 text2前`j`个字符的最长公共子序列的长度。即`text1[0....i-1]`和`text2[0...j-1]`的最长公共子序列的长度。则可得边界条件，当`i=0`或`j=0`时即其中有一个为空字符,`dp[0][j]=dp[i][0]=0`

当`i>0`且`j>0`时：

- 当`text1[i-1] = text[j -1]`时，将这两个相同的字符称为公共字符，考虑`text1[0 : i-1]`和`text2[0 : j-1]`的最长公共子序列，再增加一个字符(即公共字符）即可得到`text1[0 : i]`和`text2[0 : j]`的最长公共子序列，因此`dp[i][j]= dp[i- 1][j-1]＋1`。
- 当`text1[i-1]≠ text2[j -1]`时，考虑以下两项:
  - `text1[0 : i-1]`和`text2[0 :j-2]`的最长公共子序列，即`dp[i][j-1]`
  - `text1[0 :i-2]`和`text2[0 : j-1]`的最长公共子序列，即`dp[i-1][j]`

要得到`text1 [0 : i-1]`和`text2[0 :j-1]`的最长公共子序列，应取两项中的长度较大的一项，因此 `dp[i][j]= max( dp[i- 1][j], dp[i][j-1])`。

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m=text1.size(),n=text2.size();
        //dp[i][j]:text1前i个字符[0~i-1]与text2前j个字符[0~j-1]的最长公共子序列长度
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++){
                if(text1[i-1]==text2[j-1]) dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        return dp[m][n];
    }
};
```

### 72.编辑距离

给你两个单词 `word1` 和 `word2`， *请返回将 `word1` 转换成 `word2` 所使用的最少操作数* 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

 

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

 

**提示：**

- `0 <= word1.length, word2.length <= 500`
- `word1` 和 `word2` 由小写英文字母组成

**思路：**

- `f(i,j)`表示，集合为所有将第一个字符串前`i`个字符变为第二个字符串前`j`个字符的方式的最少操作数量
- 集合划分：以第一个字符串`i`处可能进行的三种不同操作后转化为第二个字符串。
  - **删去**第`i`个字符，即前`i-1`个字符已经与第二个字符串的前`j`个字符相同，因此只需要在上一个状态加上删去操作即可，即`f(i,j)=f(i-1,j)+1`；
  - 第`i`个字符后面**增加**一个字符，即第一个字符串前`i`个字符已经与第二个字符串的前`j-1`个字符相同，需要在第一个字符串的末尾加上一个字符，因此只需要在上一个状态上加上插入操作即可，即`f(i,j)=f(i,j-1)+1`；
  - **修改**第`i`个字符，即前`i-1`个字符已经与第二个字符串的前`j-1`个字符相同，再比较第i个字符是否与第j个字符相同，若相同就不用操作，若不同则需要增加一次修改操作,即`f(i,j)=f(i-1,j-1)+0 or 1`。
- 最终`f(i,j)`取三者最小值

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m=word1.size(),n=word2.size();
        //dp[i][j]表示word1前i个字符变换到word2前j个字符的最少操作数
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        //word2为空字符 则word1一直删除
        for(int i=1;i<=m;i++) dp[i][0]=i;
        //word1为空字符 则已知插入一个字符
        for(int i=1;i<=n;i++) dp[0][i]=i;

        for(int i=1;i<=m;i++)
            for(int j=1;j<=n;j++){
                //删除或插入操作 取操作数小者
                dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1);
                //最后一个字符不相等，则还要加入一个修改操作
                if(word1[i-1]!=word2[j-1]) dp[i][j]=min(dp[i][j],dp[i-1][j-1]+1);
                //否则不用修改操作
                else dp[i][j]=min(dp[i][j],dp[i-1][j-1]);
            }
        return dp[m][n];
    }
};
```

## 技巧

### 136.只出现一次的数字

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

 

**示例 1 ：**

```
输入：nums = [2,2,1]
输出：1
```

**示例 2 ：**

```
输入：nums = [4,1,2,1,2]
输出：4
```

**示例 3 ：**

```
输入：nums = [1]
输出：1
```

 

**提示：**

- `1 <= nums.length <= 3 * 104`
- `-3 * 104 <= nums[i] <= 3 * 104`
- 除了某个元素只出现一次以外，其余每个元素均出现两次。

**思路：**

对于这道题，可使用异或运算 ⊕。异或运算有以下三个性质。

- 任何数和 0 做异或运算，结果仍然是原来的数，即 $a⊕0=a$。

- 任何数和其自身做异或运算，结果是 0，即 $a⊕a=0$。
- 异或运算满足交换律和结合律，即 $a⊕b⊕a=b⊕a⊕a=b⊕(a⊕a)=b⊕0=b$。

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res=0;
        //遍历做异或运算，留下的即为只出现一次的数
        for(auto& num:nums){
            res^=num;
        }
        return res;
    }
};
```

### 169.多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

**示例 1：**

```
输入：nums = [3,2,3]
输出：3
```

**示例 2：**

```
输入：nums = [2,2,1,1,1,2,2]
输出：2
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 5 * 104`
- `-109 <= nums[i] <= 109`

**思路：**本题相当于找众数，三种解法

- 哈希表统计法： 遍历数组 nums ，用 HashMap 统计各数字的数量，即可找出 众数 。此方法时间和空间复杂度均为 O(N) 。

- 数组排序法： 将数组 nums 排序，数组中点的元素 一定为众数。
- 摩尔投票法： 核心理念为 票数正负抵消 。此方法时间和空间复杂度分别为 O(N) 和 O(1) ，为本题的最佳解法。

**摩尔投票法**：

- 推论一： 若记 众数 的票数为 +1 ，非众数 的票数为 −1 ，则一定有所有数字的 票数和 >0 。

- 推论二： 若数组的前 a 个数字的 票数和 =0 ，则 数组剩余 (n−a) 个数字的 票数和一定仍 >0 ，即后 (n−a) 个数字的 众数仍为 x 。

根据以上推论，记数组首个元素为 n ，众数为 x ，遍历并统计票数。当发生 票数和 =0 时，剩余数组的众数一定不变 ，这是由于：前面和非众数和众数的数量是相等的，所以抵消为0了。


利用此特性，每轮假设发生 票数和 =0 都可以 缩小剩余数组区间，此时重新设置一下当前的众数 。当遍历完成时，最后一轮假设的数字即为众数。

**实现：**

- 初始化： 票数统计 votes = 0 ， 众数 x。
- 循环： 遍历数组 nums 中的每个数字 num 。
  - 当 票数 votes 等于 0 ，则假设当前数字 num 是众数。
  - 当 num = x 时，票数 votes 自增 1 ；当 num != x 时，票数 votes 自减 1 。
- 返回值： 返回 x 即可。

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        //众数、票数
        int res=0,votes=0;
        for(auto& num:nums){
            //前一轮得到的票数和为0，则把当前数看做众数
            if(votes==0) res=num;
            //计算本轮票数和
            if(num==res) votes++;//当前数与众数相等 票数+1
            else votes--;//不等 票数-1
        }
        return res;
    }
};
```

### 75.颜色分类

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**[原地](https://baike.baidu.com/item/原地算法)** 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

 

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```
输入：nums = [2,0,1]
输出：[0,1,2]
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 300`
- `nums[i]` 为 `0`、`1` 或 `2`

**思路：**

方法一：双指针

考虑使用指针**p0来交换0，p2来交换2**。此时，**p0的初始值仍然为0，而p2的初始值为n -1**。在遍历的过程中，我们需要找出所有的0交换至数组的头部，并且找出所有的2交换至数组的尾部。

由于此时其中一个指针`p2`是从右向左移动的，因此当我们在从左向右遍历整个数组时，如果遍历到的位置超过了`p2`，那么就可以直接停止遍历了。

具体地，我们从左向右遍历整个数组，设当前遍历到的位置为`i`，对应的元素为`nums[i]`

- 如果找到了2，那么将其与`nums[p2]`进行交换，并将`p2`向前移动一个位置

  注意：对于这种情况，当我们将`nums[i]`与`num[p2]`进行交换之后，交换后的`nums[i]`可能仍然是2，也可能是0。然而此时我们已经结束了交换，开始遍历下一个元素`nums[i+1]`，不会再考虑`nums[i]`了，那这个可能为2的`nums[i]`就永远的停留于此。

  因此，当我们找到2时，我们需要**不断**地将其与`nums[p2]`进行交换，直到新的`nums[i]`不为2。

- 如果找到了0，将其与`nums[p0]`进行交换，并将`p0`向后移动一个位置;

- 如果`nums[i]`为1，那么就不需要进行任何后续的操作。

```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int p0=0,p2=nums.size()-1;//分别指明当前要存放0和2的位置
        for(int i=0;i<=p2;i++){//遍历到p2的位置为止
            while(i<=p2 && nums[i]==2){//nums[i]为2的话就不断交换 就是把当前的2放到数组末尾
                swap(nums[i],nums[p2]);
                p2--;//位置前移
            }
            if(nums[i]==0){//为0 就放到适当位置
                swap(nums[i],nums[p0]);
                p0++;//位置后移
            }
        }
    }
};
```

方法二：刷油漆法[75. 颜色分类 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-colors/?envType=study-plan-v2&envId=top-100-liked)

- 遍历过程中，先保存当前值，然后将当前位置值置为2；
- 然后判断当前值是否小于2，合适位置置1
- 然后再判断当前值是否为0，合适位置置0

类似于先全部刷2，再刷上1，再在前面的位置刷上0；

```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        //n0记录0的个数，n1记录0和1的个数 为后来选择适当的位置放置
        int n0=0,n1=0;
        for(int i=0;i<nums.size();i++){
            int num=nums[i];//记录当前值
            nums[i]=2;//然后不管三七二十一将当前值置为2
            if(num<2) nums[n1++]=1;//当前值小于2(0或1)，计数+1，对应位置置1
            if(num<1) nums[n0++]=0;//再看当前值是否为0，为0，计数+1，在合适的位置置0
        }
        //
    }
};
```

### 31.下一个排列

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须**[ 原地 ](https://baike.baidu.com/item/原地算法)**修改，只允许使用额外常数空间。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

 

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 100`

**思路：**

**推导**

1. 我们希望下一个数 **比当前数大**，这样才满足 “下一个排列” 的定义。因此只需要 **将后面的「大数」与前面的「小数」交换**，就能得到一个更大的数。比如 123456，将 5 和 6 交换就能得到一个更大的数 123465。
2. 我们还希望下一个数 **增加的幅度尽可能的小**，这样才满足“下一个排列与当前排列紧邻“的要求。为了满足这个要求，我们需要：
   1. 在 **尽可能靠右的低位** 进行交换，需要 **从后向前** 查找
   2. 将一个 **尽可能小的「大数」** 与前面的「小数」交换。比如 123465，下一个排列应该把 5 和 4 交换而不是把 6 和 4 交换
   3. 将「大数」换到前面后，需要将「大数」后面的所有数 **重置为升序，升序排列就是最小的排列**。以 123465 为例：首先按照上一步，交换 5 和 4，得到 123564；然后需要将 5 之后的数重置为升序，得到 123546。显然 123546 比 123564 更小，123546 就是 123465 的下一个排列

实现：

1. 从后向前 查找第一个 相邻升序 的元素对 `(i,j)`，满足 `A[i] < A[j]`。此时 `[j,end)` 必然是降序

2. 在 `[j,end) `从后向前 查找第一个满足 `A[i] < A[k] `的` k`。`A[i]、A[k] `分别就是上文所说的「小数」、「大数」
3. 将 `A[i]` 与 `A[k] `交换
4. 可以断定这时` [j,end)` 必然是降序，逆置` [j,end)`，使其升序
5. 如果在步骤 1 找不到符合的相邻元素对，说明当前 `[begin,end) `为一个降序顺序，则直接跳到步骤 4

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n=nums.size();
        for(int i=n-2;i>=0;i--){//从后向前遍历 i指向第一个升序对的第一个元素
            if(nums[i]<nums[i+1]){//找到一个小数num[i]
                //然后在i的后面再找一个大数 可能是i+1(最差)，也可能不是
                for(int k=n-1;k>i;k--){//依旧是从后向前
                    if(nums[k]>nums[i]){//找到了
                        swap(nums[i],nums[k]);//和i所在的数交换
                        //然后将i后面的数，即i+1开始的数逆置
                        reverse(nums.begin()+i+1,nums.begin()+n);
                        //reverse函数翻转下标[i+1,n)内的数
                        return ;//ok了提前退出
                    }
                }
            }
        }
        //如果没有提前退出 则原本数组就是一个完全降序的数组 直接逆置
        reverse(nums.begin(),nums.end());
    }
};
```

### 287.寻找重复数

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

 

**示例 1：**

```
输入：nums = [1,3,4,2,2]
输出：2
```

**示例 2：**

```
输入：nums = [3,1,3,4,2]
输出：3
```

**示例 3 :**

```
输入：nums = [3,3,3,3,3]
输出：3
```

 

**提示：**

- `1 <= n <= 105`
- `nums.length == n + 1`
- `1 <= nums[i] <= n`
- `nums` 中 **只有一个整数** 出现 **两次或多次** ，其余整数均只出现 **一次**

**思路**：二分查找

每一次猜一个数，然后 遍历整个输入数组，进而缩小搜索区间，最后确定重复的是哪个数。

理解题意：

- `n + 1` 个整数，放在长度为` n `的数组里，根据「抽屉原理」，至少会有 1 个整数是重复的；

  > [抽屉原理](https://leetcode.cn/link/?target=https%3A%2F%2Fbaike.baidu.com%2Fitem%2F抽屉原理%2F233776)：把 `10` 个苹果放进 `9` 个抽屉，至少有一个抽屉里至少放 `2` 个苹果。

「二分查找」的思路是先猜一个数（搜索范围` [left..right] `里位于中间的数` mid`），然后统计原始数组中 **小于等于** `mid `的元素的个数 `count`：

- 如果 `count` 严格大于 `mid`。根据 抽屉原理，重复元素就在区间 `[left..mid] `里；

  > 如果遍历一遍输入数组，统计小于 **等于** `4` 的元素的个数，如果小于等于 `4` 的元素的个数 **严格** 大于 `4` ，说明重复的元素一定出现在整数区间 `[1..4]`（抽屉原理）

- 否则，重复元素可以在区间 `[mid + 1..right] `里找到，其实只要第 1 点是对的，这里肯定是对的

题目中说：长度为 `n + 1 `的数组，数值在 `1 `到 `n` 之间。因此长度为 `len = n + 1`，`n = len - 1`，搜索范围在 `1` 到 `len - 1 `之间；

在 循环体内，先猜一个数 `mid`，然后遍历「输入数组」，统计小于等于 `mid `的元素个数 `count`，如果 `count > mid` 说明重复元素一定出现在 `[left..mid]` 因此设置` right = mid`；

```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int len=nums.size();//n=len-1,数的范围为[1,len-1]

        //在范围[1...len-1]查找nums的重复元素
        int left=1;
        int right=len-1;
        while(left<right){//left和right均表示数的大小而不是数组下标
            int mid=(left+right)/2;//先猜一个中间数

            //看nums中有多少个小于等于mid的数
            int count=0;
            for(auto& num:nums){
                if(num<=mid) count++;
            }
            if(count>mid)//必然有重复数在左区间
                right=mid;
            else//否则在右区间
                left=mid+1;
            //然后继续下一轮搜索
        }
        return left;//或者返回right 
    }
};
```

