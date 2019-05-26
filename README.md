# Leetcode.DataStructure-notes
Leetcode之分类刷题之数据结构部分~

## No->1.数组与矩阵↓↓↓↓↓↓↓↓↓↓

**109.83. Move Zeroes (Easy)：1.把数组中的0移到末尾**
```
例如，给定nums = [0,1,0,3,12]，在调用函数后，nums应为[1,3,12,0,0]
为了干正事：放0在最后永远先异常 if(nums[i] != 0){  
```
```
public void moveZeroes(int[] nums){
	if(nums==null || nums.length==0)
		return;
	int index = 0;

	//**所有的void就是直接动自己本身
	for(int i = 0;i < nums.length;i++){
		if(nums[i] != 0){
			nums[index++] = nums[i];
			}
		}
	for(int i = index;i < nums.length;i++){
		nums[i] = 0;
	}
}
```

**110.566. Reshape the Matrix (Easy)：2. 改变矩阵维度**
```
Input:
nums =
[[1,2],
 [3,4]]
r = 1, c = 4
Output:
[[1,2,3,4]]
>行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。
Input:
nums = 
[[1,2],
 [3,4]]
r = 2, c = 4
Output:
[[1,2],
 [3,4]]
解释:
没有办法将 2 * 2 矩阵转化为 2 * 4 矩阵。 所以输出原矩阵
```

* 1.将二维的转成一维的  每遍历到一个数则index++
* 2.将一维的转成二维的  = m[cur/c,cur%c];

* if(m * n != r * c)	res[i][j] = nums[index/c][index%c];	
```
public int[][] matrixReshape(int[][] nums, int r, int c) {
	int m = nums.length;
	int n = nums[0].length;
	if(m*n != r*c)	//*这个异常好
		return nums;
	int[][] res = new int[r][c];
	int index = 0;
	for(int i = 0;i < r;i++){
		for(int j = 0;j < c;j++){
			res[i][j] = nums[index/c][index%c];
			index++;
		}
	}
	
	return res;
}
```


**111.485. Max Consecutive Ones (Easy)：3. 找出数组中最长的连续1的个数**
```
输入: [1,1,0,1,1,1] 输入的数组只包含 0 和1。
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.
```
* max = Math.max(max,counts);

```
public int findMaxConsecutiveOnes(int[] nums) {
	if(nums==null || nums.length==0)
		return -1;
	int max = 0;
	int counts = 0;
	for(int i = 0;i < nums.length;i++){
		if(nums[i]==1)
			counts++;
		else
			counts = 0;
		max = Math.max(max,counts);		//*这就是精华咯
	}
	return max;
}
```

**112.240. Search a 2D Matrix II (Medium)：4. 有序矩阵查找**
```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。给定 target = 20，返回 false。
```
```
public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) 
		return false;
    int m = matrix.length, n = matrix[0].length;
    int row = 0, col = n - 1;
    while (row < m && col >= 0) {
        if (target == matrix[row][col]) return true;
		//*没啥
        else if (target < matrix[row][col]) col--;
        else row++;
    }
    return false;
}
```


  
**113.378. Kth Smallest Element in a Sorted Matrix ((Medium))：5. 有序矩阵的 Kth Element**
```
给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第k小的元素。
请注意，它是排序后的第k小元素，而不是第k个元素。
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,返回 13
```
* k;PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>
	(k,new Comparator<Integer>{ }); maxHeap.add; maxHeap.poll();
* 任何第k怎么样的-->都可以用堆解法
* -->第k小所以是最大堆 每次把k个中最大的丢掉啦
  
```
public int kthSmallest(int[][] matrix, int k) {
	int r = matrix.length;
	int c = matrix[0].length;
	if(matrix==null || matrix.length==0)
		return -1;
	//*用优先队列 创建一个最大堆
	PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>
	(k,new Comparator<Integer>{
		public int compare(Integer o1,Integer o2){
			return o2 - o1;		//就是降序
		}
		
	});
	
	for(int i = 0; i < r;i++){
		for(int j = 0;j < c;j++){
			if(maxHeap.size() < k){
				//*无push方法
				maxHeap.add(matrix[i][j]);
			}else{
				if(matrix[i][j] < maxHeap.peek()){
					maxHeap.poll();
					maxHeap.add(matrix[i][j]);
				}
			}
		}
	}
	
	return maxHeap.peek();
}
```


**114.645. Set Mismatch (Easy)：6. 一个数组元素在 [1, n] 之间，**
```
其中一个数被替换为另一个数，找出重复的数和丢失的数
集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，
导致集合里面某一个元素复制了成了集合里面的另外一个元素的值，
导致集合丢失了一个整数并且有一个元素重复。
输入: nums = [1,2,2,4]
输出: [2,3]
```
* 因为1就对应的1 所以就好交换啦~
* 主要通过交换元素 把数组上的元素放在正确的位置   
* while(nums[index] != nums[i] return new int[]{nums[i], i + 1};    
* 放位置的时候 先找正确的1.index 2.i当前的是否等 就交换咯~

```
private void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}
public int[] findErrorNums(int[] nums) {
	int index = 0;
	if(nums==null)
		return null;
	
	//*-->通过比如2就应该放在下标为(2-1)的位置
	for(int i = 0;i < nums.length;i++){
		//*index：为当前i的值应该放的地方
		index = nums[i] - 1;
		while(nums[index] != nums[i] && nums[i] != (i+1)){
			swap(nums,i,index);
		}
	}
	//*-->找重复的数nums[i]和丢失的i+1
	//**0位置不为1 1位置不为2 2位置不为3
	for(int i = 0;i < nums.length;i++){
		if(nums[i] != (i+1)){
			return new int[]{nums[i], i + 1};
		}
	}
	return null;
}
```

**115.287. Find the Duplicate Number (Medium)：7.找出数组中重复的数，值在[1 , n]之间**
```
给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），
可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
```
* if(count > mid){     
* 找n+1的一半的下标为mid  然后用mid分收集左一半 如果左num>mid 则是较小的一部分有重复 
```
public int findDuplicate(int[] nums) {
	if(nums==null)
		return -1;
	int left = 0;
	int right = nums.length-1;
	//*更干净的遍历了
	while(left <= right){
		int mid = left + (right - left)/2;
		int count = 0;
		for(int i = 0;i < num.length;i++){
			if(nums[i] <= mid)		//找小的数量为count
				count++;
		}
		if(count > mid){	//*-->较少的一部分多
			//*因为外面的是= 所以里面一定是— + 1
			right = mid - 1;
		}else{
			left = mid + 1;
		}
	}
	return left;
}

```

**116.667. Beautiful Arrangement II (Medium)：8. 数组相邻差值的个数**
```
题目描述：数组元素为 1~n 的整数，要求构建数组，使得相邻元素的差值不相同的个数为 k。
输入: n = 3, k = 1
输出: [1, 2, 3]
解释: [1, 2, 3] 包含 3 个范围在 1-3 的不同整数， 
并且 [1, 1] 中有且仅有 1 个不同整数 : 1
输入: n = 3, k = 2
输出: [1, 3, 2]
解释: [1, 3, 2] 包含 3 个范围在 1-3 的不同整数， 
并且 [2, 1] 中有且仅有 2 个不同整数: 1 和 2
```
* -->注意k为2 的时候差值是(2,1) 所以差值是：k,k-1,k-2,...,1   
* 首先关键的一点k最多是n-1 这n-1就是由较大的数和较小的数交替形成的就满足啦~
例如n=6 k=5 则615243 or 162534 如果k=4 则615243 就是满足了3个差值(也就是i=4的数形成了)
是交替形成的之后就不管啦~    
* for(int i = 1,temp = k;i <= temp;i++,temp--){ for(int i = k + 1;i < n;i++){

```
public int[] constructArray(int n, int k) {
	if(n <= 0 || k >= n)
		return null;
	int[] res = new int[n];
	res[0] = 1;		//162534的1
	for(int i = 1,temp = k;i <= temp;i++,temp--){	//*满足k的所有哦~
		if(i%2==1){		//*第1,3,5位置上的也就是6,5啥的也就是添加咯~
			res[i] = res[i-1] + temp;
		}else{
			res[i] = res[i-1] - temp;
		}
	}
	//*比如k=4只管3个差值4个数 则i=5开始
	for(int i = k + 1;i < n;i++){
		res[i] = i+1;
	}
	return res;
}
```

**117.697. Degree of an Array (Easy)：9. 数组的度**
```
Input: [1,2,2,3,1,4,2]
Output: 6
题目描述：数组的度定义为元素出现的最高频率，例如上面的数组度为 3。
输出一个最小的子数组的长度，这个子数组的度和原数组一样。
```
* 很简单：HashMap统计数量用

```
class Solution {
    public int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> left = new HashMap();
        Map<Integer, Integer> right = new HashMap();
        Map<Integer, Integer> count = new HashMap();
        
        for(int i=0; i < nums.length; i++ ){
			//*没有存在过
            if(!left.containsKey(nums[i]))
                left.put(nums[i], i);
				right.put(nums[i], i);
				count.put(nums[i], count.getOrDefault(nums[i],0)+1);
        }
        
		//*调用Collections.max系统函数 取得最大值的度
        int degree = Collections.max(count.values());
        int length = Integer.MAX_VALUE;
        for(int i=0; i<nums.length; i++){
            if(count.get(nums[i])==degree){
                length = Math.min(length, right.get(nums[i]) - left.get(nums[i]) + 1);
            }
        }
        return length;
    }
}
```

**118.766. Toeplitz Matrix (Easy)：10. 对角元素相等的矩阵**
```
输入: 
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
输出: True
解释:
在上述矩阵中, 其对角线为:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。
各条对角线上的所有元素均相同, 因此答案是True。
```
* if(matrix[i][j] != matrix[i-1][j-1])   
* 从matrix[1][1]开始遍历因为要-1的 分别和当前的左上角对比

```
public boolean isToeplitzMatrix(int[][] matrix) {
	 if(matrix==null || matrix.length==0)
		 return false;
	 for(int i = 1;i < matrix.length;i++){
		 for(int j = 1;j < matrix[0].length;j++){
			 if(matrix[i][j] != matrix[i-1][j-1])
				 return false;
		 }
	 }
	 return true;
}
```

**119.565. Array Nesting (Medium)：11. 嵌套数组**
```
Input: A = [5,4,0,3,1,6,2]
Output: 4
Explanation:
A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.
One of the longest S[K]:
S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}-->满足循环了
题目描述：假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，
之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素。
```
* 其实就是找最大环：只要环起来了并且过半就行了    
* if(max > nums.length/2) int cur = nums[nums[i]]; while(nums[i] != cur){

```
public int arrayNesting(int[] nums) {
	int res = 1;
	boolean[] visited = new boolean[nums.length];
	for(int i = 0;i < nums.length;i++){
		//访问过就不行了
		if(visited[nums[i]])
            continue;
		//*这样就可以找到啦~
		if(max > nums.length/2)
			return max;
		visited[nums[i]] = true;
		int curMax = 1;
		int cur = nums[nums[i]];
		//*还没成环 还没相遇
		while(nums[i] != cur){
			visited[cur] = true;
			curMax++;
			//*继续下一个
			cur = nums[cur];
		}
		max = curMax > max ? curMax : max;
	}
	return max;
}

```

**164.769. Max Chunks To Make Sorted (Medium)：12. 分隔数组**
```
数组arr是[0, 1, ..., arr.length - 1]的一种排列，我们将这个数组分割成几个“块”，
并将这些块分别进行排序。
再连接起来，使得连接的结果和按升序排序后的原数组相同。
我们最多能将数组分成多少块？
输入: arr = [4,3,2,1,0]
输出: 1
解释:
将数组分成2块或者更多块，都无法得到所需的结果。
例如，分成 [4, 3], [2, 1, 0] 的结果是 [3, 4, 0, 1, 2]，这不是有序的数组。
题目描述：分隔数组，使得对每部分排序后数组就为有序。
输入: arr = [1,0,2,3,4]
输出: 4
解释:
我们可以把它分成两块，例如 [1, 0], [2, 3, 4]。
然而，分成 [1, 0], [2], [3], [4] 可以得到最多的块数。
```
* 记录当前区间的max 每加入一个i   
* 如果这个max 也就是排序的最后的right了 ==下标 就是这个区间ok的 则区间+1

```
public int maxChunksToSorted(int[] arr) {
	if(arr==null)
		return 0;
	int res = 0;
	int max = 0;
	for(int i = 0;i < arr.length;i++){
		max = Math.max(max,arr[i]);
		if(max == i)
			res++;
	}
	return res;
}
```

## No->2.字符串↓↓↓↓↓↓↓↓↓↓

**120.字符串循环移位包含**
```
s1 = AABCD, s2 = CDAA
Return : true
给定两个字符串 s1 和 s2，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。
s1 进行循环移位的结果是 s1s1 的子字符串，因此只要判断 s2 是否是 s1s1 的子字符串即可。
```
```
1.s1+s1
2.s.contains(s2)
```

**121.字符串循环移位**
```
s = "abcd123" k = 3
Return "123abcd"
将字符串向右循环移动 k 位。
将 abcd123 中的 abcd 和 123 单独翻转，得到 dcba321，
然后对整个字符串进行翻转，得到 123abcd。
```
```
s1 = s.subString(0, from.length() - index)
s2 = s.subString(from.length() - index)
reverse(s1) reverse(s2) reverse(s)
```

**122.字符串中单词的翻转**
```
s = "I am a student"
Return "student a am I"
```
* 将每个单词翻转，然后将整个字符串翻转

```
String[] str = sentence.split(" ")
StringBuffer sb = new StringBuffer();
for (int i = str.length-1; i >=0; i--) {
		sb.append(str[i]+" ");
}
```

**123.242. Valid Anagram (Easy)：两个字符串包含的字符是否完全相同**
```
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.
可以用 HashMap 来映射字符与出现次数，然后比较两个字符串出现的字符数量是否相同。
由于本题的字符串只包含 26 个小写字符，
因此可以使用长度为 26 的整型数组对字符串出现的字符进行统计，不再使用 HashMap。
```
* counts[]就可以啦
```
public boolean isAnagram(String s, String t) {
    int[] counts = new int[26];
	char[] cs1 = s.toCharArray();
	char[] cs2 = t.toCharArray();
    for (char c : cs1) {
		//*c-'a'很秀啊
        counts[c - 'a']++;
    }
    for (char c : cs2) {
        counts[c - 'a']--;
    }
    for (int count : counts) {
        if (count != 0) {
			//*中间凉就凉
            return false;
        }
    }
    return true;
}
```

**124.409. Longest Palindrome (Easy)：计算一组字符集合可以组成的回文字符串的最大长度**
```
Input : "abccccdd"
Output : 7
Explanation : "dccaccd", whose length is 7.
```
* 使用长度为 256 的整型数组来统计每个字符出现的个数，每个字符有偶数个可以用来构成回文字符串。  
* 因为回文字符串最中间的那个字符可以单独出现，所以如果有单独的字符就把它放到最中间。    
* 可以组成的-->这个是可以打乱的 所以不能双指针
* 我们可以统计出每个的count only偶数的可以所以count/2 * 2就去小数了  再+1奇数   
* res = res + (count/2) * 2;
```
public int longestPalindrome(String s) {
	//*256 和 26的不同
	if(s==null)
		return -1;
	int[] counts = new int[256];
	char[] cs = s.toCharArray();
	for(char c : cs){
		//*还是有什么！？之类的字符串的
		counts[c]++;
	}
	int res = 0;
	for(int count : counts){
		res = res + (count/2)*2;
	}
	if(res < s.length())	//*加个中点
		res++;
	return res;
}
```

**125.647. Palindromic Substrings (Medium)：回文子字符串个数**
```
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。
输入: "abc"
输出: 3
解释: 3个回文子串: "a", "b", "c".
输入: "aaa"
输出: 6
说明: 6个回文子串: "a", "a", "a", "aa", "aa", "aaa".
```
* 从字符串的某一位开始，尝试着去扩展子字符串。   
* 如果是奇数长度，那么i位置就是中间那个字符的位置，所以我们左右两遍都从i开始遍历    
* 如果是偶数长度的，那么i是最中间两个字符的左边那个，右边那个就是i+1   
* extendString(s,i,i); extendString(s,i,i+1);    
```
int counts = 0;

private void extendString(String s,int left,int right){
	char[] cs = s.toCharArray();
	//*双指针就是while
	while(left>=0 && right<=(s.length()-1) && cs[left]==cs[right]){
		counts++;
		left--;
		right++;
	}
}

public int countSubstrings(String s){
	for(int i = 0;i < s.length();i++){
		//*因为不同的起点的哪怕都是a都算两种啦~
		extendString(s,i,i);
		extendString(s,i,i+1);
	}
	return counts;
}

```

**126.9. Palindrome Number (Easy)：判断一个整数是否是回文数**  

```
要求不能使用额外空间，也就不能将整数转换为字符串进行判断。
将整数分成左右两部分，右边那部分需要转置，然后判断这两部分是否相等。
输入: 121
输出: true
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```
* 偶数长：则left == right 奇数长：则右长 则left == right/10;   
* left%10 则搞出左边的给右边的加到个位的 rigth 为右边的值 * 10是为了翻转   
* -->总之 left用/ right用%   
* right = right * 10 + lastNum; return left==right || left==right/10;    
```
public boolean isPalindrome(int x) {
	if(x==0)
		return true;
	//*因为数不可能以0开头所以0结尾则凉凉
	if(x < 0 || x % 10 == 0)
		return false;
	int right = 0;
	int left = x;
	int lastNum = 0;	//当前左的最后一位数
	while(left > right){
		//*1.锤出当前左的最后一位数
		lastNum = left%10;
		//***2.翻转当前的新右 原来的右*10打头 
		right = right*10 + lastNum;
		//*3.继续锤新left
		left = left/10;
	}
	return left==right || left==right/10;
}
```

**127.696. Count Binary Substrings (Easy)**

``` 
统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数
输入: "00110011"
输出: 6
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
```
* pre为上一个数字的个数 cur为当前数字的个数   
* 如果pre>=cur 则肯定满足连续1和连续0的相同个数 不管是0在前还是1在前    
```
public int countBinarySubstrings(String s) {
	if(s==null)
		return -1;
	int pre = 0, cur = 1,res = 0;
	char[] cs = s.toCharArray();
	for(int i = 1;i < s.length();i++){
		if(cs[i]==cs[i-1])
			cur++;
		else{
			pre = cur;
			cur = 1;
		}
		if(pre >= cur)
			res++;
	}
	return res;
}
```

## No->3.树↓↓↓↓↓↓↓↓↓↓

### 3.1：递归↓↓↓↓↓↓↓↓↓↓
* 树本身就是一种递归结构
一棵树要么是空树，要么有两个指针，每个指针指向一棵树。   
* 深度题：return Math.max(Depth(root.left),Depth(root.right))+1;    
* 用递归的地方：.left = 递归root.left .right = 递归root.right 因为这两个都是树哦~    

**128.104. Maximum Depth of Binary Tree (Easy)：树的高度**
```
public int maxDepth(TreeNode root) {
	if(root==null)
		return 0;
	return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
}
```

**129.110. Balanced Binary Tree (Easy)：平衡树**  
```
    3     
   / \    
  9  20     
    /  \    
   15   7 平衡树左右子树高度差都小于等于 1   
```
 
* 所有的子树也得ok的!
```
private int Depth(TreeNode root){
	if(root==null)
		return 0;
	return Math.max(Depth(root.left),Depth(root.right))+1;
}
public boolean isBalanced(TreeNode root){
	if(root==null)
		return true;
	int l = Depth(root.left);
	int r = Depth(root.right);
	if(Math.abs(l-r) > 1)
		return false;
	//*这里还需要递归子的~
	return isBalanced(root.left)&&isBalanced(root.right);
}
```

**130.543. Diameter of Binary Tree (Easy)：两节点的最长路径**  
```
Input:       
        
         1       
        / \        
       2  3          
      / \         
     4   5          
        
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3]
```

* -->左右深度相加就行  
* 深度题：return Math.max(Depth(root.left),Depth(root.right))+1;    
* -->对于有递归都要用则全局变量~     
```
int res = 0;

public int diameterOfBinaryTree(TreeNode root) {
	if(root==null)
		return 0;
	helper(root);
	return res;
}

private int helper(TreeNode root){
	if(root==null)
		return 0;	//*用来求深度的
	int lDep = helper(root.left);
	int rDep = helper(root.right);
	int cur = lDep + rDep;
	res = Math.max(res,cur);
	return Math.max(lDep,rDep)+1;
}
```

**131.226. Invert Binary Tree (Easy)：翻转树**
```
public TreeNode invertTree(TreeNode root) {
	if(root==null)
		return null;
	TreeNode temp = root.left;	//*保存给下面交换用
	//.left为右传递进去哦~
	root.left = invertTree(root.right);
	root.right = invertTree(temp);
	return root;
}
```
* 非递归版本，其实也不难。如果你会非递归的遍历树那么这个是一个道理   
* -->非递归也就是栈啦   
```
public TreeNode invertTree(TreeNode root) {
	if(root==null)
		return null;
	Stack<TreeNode> s = new Stack<>();
	s.push(root);
	while(!s.isEmpty()){
		TreeNode cur = s.pop();
		//*处理交换逻辑：
		TreeNode temp = cur.left;
		cur.left = cur.right;
		cur.right = temp;
		//*中序遍历~继续放
		if(cur.left!=null)
			s.push(cur.left);
		if(cur.right!=null)
			s.push(cur.right);
	}
	return root;
}
```

**132.617. Merge Two Binary Trees (Easy)：归并两棵树**
```
Input:    
       Tree 1                     Tree 2        
          1                         2          
         / \                       / \          
        3   2                     1   3         
       /                           \   \           
      5                             4   7           
      
Output:         
         3         
        / \        
       4   5          
      / \   \          
     5   4   7        

public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
	if(t1==null && t2==null)
		return null;
	if(t1==null)
		return t2;
	if(t2==null)
		return t1;
	//*逻辑
	TreeNode root = new TreeNode(t1.val + t2.val);
	//*继续递归
	root.left = mergeTrees(t1.left,t2.left);
	root.right = mergeTrees(t1.right,t2.right);
	return root;
}
```

**133.112. Path Sum (Easy)：必须以root开始的路径和是否等于一个数**
> Given the below binary tree and sum = 22      

              5      
             / \    
            4   8        
           /   / \         
          11  13  4      
         /  \      \        
        7    2      1       

return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22    
* 大局观：路径和定义为从 root 到 leaf 的所有节点的和
```
public boolean hasPathSum(TreeNode root, int sum) {
	if(root==null)
		return false;
	//*2.1：跳出递归OK了
	if(root.left==null && root.right==null && root.val==sum)
		return true;
	//*2.2：继续递归的情况~ 维护下新sum
	sum = sum - root.val;
	//*-->依旧是左右孩子的递归
	return hasPathSum(root.left,sum) || hasPathSum(root.right,sum);
}
```

**134.437. Path Sum III (Easy)：统计路径和等于一个数的路径数量**
```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
路径不一定以 root 开头，也不一定以 leaf 结尾，但是必须连续。
Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

* 1.以当前头开始走~ 2.以.left递归 3.以.right递归   

```

```

**135.572. Subtree of Another Tree (Easy)：子树**
```
public boolean isSubtree(TreeNode s, TreeNode t) {
	if(s==null)
		return false;
	//*1.大局观：三部分
	return startWithRoot(s,t) || isSubTree(s.left,t) || isSubTree(s.right,t);
}

private boolean startWithRoot(TreeNode s,TreeNode t){
	//*2.1：跳出递归的情况
	if(s==null && t==null)
		return true;
	if(s==null || t==null)
		return false;
	if(s.val != t.val)
		return false;
	//*2.2：继续递归的情况~
	return startWithRoot(s.left,t.left) && startWithRoot(s.right,t.right);
}
```

**137.111. Minimum Depth of Binary Tree (Easy)：最小路径**
```
    3
   / \					  
  9  20         
    /  \			
   15   7			
返回它的最小深度  2.
```
* 1.找lDep rDep 2.结果就是：~ + 1
```
public int minDepth(TreeNode root) {
	
	if(root==null)
		return 0;
	//*方框用递归~
	int lDep = minDepth(root.left);
	int rDep = minDepth(root.right);
	//*方便处理罢了~
	if(lDep==0 || rDep==0)
		return (lDep + rDep + 1);
	//*都在
	return Math.min(lDep,rDep) + 1;
}
```

**136.101. Symmetric Tree (Easy)：树的对称**
```
   	1
   / \
  2   2
 / \ / \
3  4 4  3
```
* 一变二
```
public boolean isSymmetric(TreeNode root) {
	if(root==null)
		return false;
	return isSymmetric(root,root);
}
private boolean isSymmetric(TreeNode s,TreeNode t){
	if (t1 == null && t2 == null) return true;
    if (t1 == null || t2 == null) return false;
	//*-->一定是异常在前，false了就不谈了 true继续递归~
    if (t1.val != t2.val) return false;
    return isSymmetric(t1.left, t2.right) && isSymmetric(t1.right, t2.left);
}
```

**138.404. Sum of Left Leaves (Easy)：统计左叶子节点的和**
```
    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively.
Return 24.
```
* 判断节点是否是左叶子节点，如果是则将它的和累计起来
```
public int sumOfLeftLeaves(TreeNode root) {
	if(root==null)
		return 0;
	int res = 0;
	//*0.处理递归真实的逻辑
	if(root.left!=null && root.left.left==null && root.left.right==null)
		res = res + root.left.val;
	
	//*1.大局观：res是根节点的左 + 然后递归新跟
	res = res + sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
	return res;
}
```

**139.687. Longest Univalue Path (Easy)：相同节点值的最大路径长度**
```
给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 
这条路径可以经过也可以不经过根节点。
注意：两个节点之间的路径长度由它们之间的边数表示。
输入:

              5
             / \
            4   5
           / \   \
          1   1   5
输出:

2 为5-5-5的路径长度
```
* 调用递归的~可以用全局的res
```
int res = 0;

public int longestUnivaluePath(TreeNode root) {
	helper(root);
	return res;
}

private int helper(TreeNode root){
	if(root==null)
		return 0;
	//*先加了一条路径啦~
	int left = helper(root.left) + 1;
	int right = helper(root.right) + 1;
	if(root.left!=null && root.left.val!=root.val)
		left = 0;
	if(root.right!=null && root.right.val!=root.val)
		right = 0;
	//*当前这样的结果
	res = Math.max(res,left + right);
	//*返回的是当前路径的个数的max
	return Math.max(left,right);
}
```

**140.337. House Robber III (Medium)：间隔遍历：打家劫舍 III**
```
除了“根”之外，每栋房子有且只有一个“父“房子与之相连。
一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
```
* 要不然135 要不然246
```
public int rob(TreeNode root) {
	if(root==null)
		return 0;
	int val1 = root.val;
	if(root.left!=null)
		val1 = val1 + rob(root.left.left) + rob(root.left.right);
	if(root.right!=null)
		val1 = val1 + rob(root.right.left) + rob(root.right.right);
	int val2 = rob(root.left) + rob(root.right);
	return Math.max(val1,val2);
}
```

**141.671. Second Minimum Node In a Binary Tree (Easy)：找出二叉树中第二小的节点**
```
给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。
如果一个节点有两个子节点的话，那么这个节点的值不大于它的子节点的值。 
Input:
   2
  / \
 2   5
    / \
    5  7

Output: 5
```
* 因为父节点比左右的小，而且左右成双成对，所以第二小在根节点的第一个不和root一样的左右的min
```
public int findSecondMinimumValue(TreeNode root) {
	if(root==null)
		return -1;
	if(root.left==null && root.right==null)
		return -1;
	int left = root.left.val;
	int right = root.right.val;
	if(left == root.val)
		left = findSecondMinimumValue(root.left);
	if(right == root.val)
		right = findSecondMinimumValue(root.right);
	if(left!=-1 && right !=-1)
		return Math.min(left,right);
	if(left!=-1)
		return left;
	  return right;
}
```



























