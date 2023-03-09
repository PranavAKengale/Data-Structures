# Palindrome Check (Easy)

string='abcdcba'

# Valid Palidrome (Leetcode Name)
# Approach 1
# T:O(n^2), S:O(n)      ....... T:O(n^2), karan string apan concat kartoy
def palindromeCheck(string):
    new=''
    for i in range(len(string)):
        new+=string[i]
    return new==string

palindromeCheck(string)

# Approach: 2
# T:O(n), S:O(n)      ....... T:O(n), karan string apan append kartoy and later join kartoy tyala O(n)ekda...so O(n)+O(n)
def palindromeCheck1(string):
    newCharacters=[]
    for i in range(len(string)):
        newCharacters.append(string[i])
    return ''.join(newCharacters)==string

palindromeCheck1(string)

# Apprach:3 ......Recursive solution
# T:O(n) , S:O(n)   .....S:O(n) karan call stack (All the function calls we have to store in call stack)
# T:O(n) , S:O(1)   .....S:O(1) he jar karaichai tr apla functions apan lasta la call karaich (Tail recursion boltat)(might be)

def isPalindrome(string,i=0):
    j=len(string)-1-i
    return True if i>=j else string[i]==string[j] and isPalindrome(string,i+1)

# OR.... Same varchya sarkha neet kalel asa(Khalch code ne tula kalel....recursive ka boltat yeila)

def isPalindrome1(string,i=0):
    j=len(string)-1-i
    if i>=j:
        return True
    if string[i]!=string[j]:
        return False
    return isPalindrome(string,i+1)

isPalindrome(string)

isPalindrome1(string)

# Apprach:4 (Optimized Approach)
# T:O(n) , S:O(1)
def palindromeCheck2(string):
    leftIdx=0
    rightIdx=len(string)-1
    while leftIdx<rightIdx:
        if string[leftIdx]!=string[rightIdx]:
            return False
        leftIdx+=1
        rightIdx-=1
    return True

palindromeCheck2(string)



# Longest Palindrome Substring (Medium)

# Longest Palindromic Substring (Leetcode name)
# T:O(n^2) S:O(n)       .............function(getLongestPalindrome) madhun everytime toh magch, pudch check karat pudhe janr...tr te pan O(n) times asnr
def longestPalindrome(string):
    currentLongest=[0,1]
    for i in range(1,len(string)):
        odd=getLongestPalindrome(string,i-1,i+1)
        even=getLongestPalindrome(string,i-1,i)
        longest=max(odd,even,key=lambda x:x[1]-x[0])
        currentLongest=max(currentLongest,longest,key=lambda x:x[1]-x[0])
    return string[currentLongest[0]:currentLongest[1]]

def getLongestPalindrome(string,leftIdx,rightIdx):
    while leftIdx>=0 and rightIdx<len(string):
        if string[leftIdx]!=string[rightIdx]:
            break
        leftIdx-=1
        rightIdx+=1
    return [leftIdx+1,rightIdx]       

string="abaxyzzyxf"

longestPalindrome(string)



# Longest Substring without Duplication (Difficult)

string="clementisaczbrap"

# Longest Substring Without Repeating Characters (Leetcode name)
# T:O(n) , S:O(min(n,a))..........S:O(min(n,a)) kalal nahi parat bag
def longestSubstringWithoutDuplication(string):
    lastseen={}
    longest=[0,1]
    startIdx=0
    for i,char in enumerate(string):
        if char in lastseen:
            startIdx=max(startIdx,lastseen[char]+1)
        if longest[1]-longest[0]< (i+1)-startIdx:
            longest=[startIdx,i+1]
        lastseen[char]=i
    return string[longest[0]:longest[1]]

longestSubstringWithoutDuplication(string)



# Caesar Cipher Encryptor (Easy)

string="xyz"
key=2

# O
def caesarCipherEncryptor(string,key):
    newLetters=[]
    newKey=key%26
    for letter in string:
        newLetter=getNewAlphabet(letter,newKey)
        newLetters.append(newLetter)
    return ''.join(newLetters)

def getNewAlphabet(letter,newKey):
    number=ord(letter) + newKey
    return chr(number) if number<=122 else chr(96+number%122)

caesarCipherEncryptor(string,key)



# One Edit

#  Leetcode Name: One Edit Distance
# Time: O(N) , Space: O(1)
def oneEdit(stringOne, stringTwo):
    lengthOne,lengthTwo=len(stringOne),len(stringTwo)
    if abs(lengthOne-lengthTwo)>1:
        return False
    madeEdit=False
    indexOne=0
    indexTwo=0

    while indexOne<lengthOne and indexTwo<lengthTwo:
        if stringOne[indexOne] != stringTwo[indexTwo]:
            if madeEdit:
                return False
            madeEdit=True

            if lengthOne>lengthTwo:
                indexOne+=1
            elif lengthOne<lengthTwo:
                indexTwo+=1
            else:
                indexOne+=1
                indexTwo+=1
        else:
            indexOne+=1
            indexTwo+=1
    return True




# Group Anagrams (Medium)

string=["yo", "act", "flop", "tac", "foo", "cat", "oy", "olfp"]

# Time: O(WNlogn), Space: O(WN)
def groupOfANagrams(string):
    hashTable={}
    for word in string:
        sortedWord=''.join(sorted(word))
        if sortedWord in hashTable:
            hashTable[sortedWord].append(word)
        else:
            hashTable[sortedWord]=[word]
    return list(hashTable.values())

groupOfANagrams(string)



# Underscorify Substring (Hard)







# Run Length Encoding (Easy)

string="AAAAAAAAAAAAABBCCCCDD"

def runLengthEncoding(string):
    encodedList=[]
    currentLength=1
    for i in range(1,len(string)):
        currentWord=string[i]
        previousWord=string[i-1]
        if currentWord!=previousWord or currentLength==9:
            encodedList.append(str(currentLength))
            encodedList.append(previousWord)
            currentLength=0
        currentLength+=1
    encodedList.append(str(currentLength))
    encodedList.append(string[(len(string)-1)])
    return ''.join(encodedList)

runLengthEncoding(string)





# Valid IP Addresses (Medium)

string="1921680"

string1="192.168@1.1"

# Leetcode name (Restore IP Addresses)
# Time: O(1), Space: O(1)       ------ Because at most there would be len 12 and looping len 12 will be a constant time operation

def validIPAdresses(string):
    ipAddressFound=[]
    for i in range(1,min(len(string),4)):
        currentIpAddress=['','','','']
        currentIpAddress[0]=string[:i]
        if not isValid(currentIpAddress[0]):
            continue
        for j in range(i+1,min(len(string),i+4)):
            currentIpAddress[1]=string[i:j]
            if not isValid(currentIpAddress[1]):
                continue
            for k in range(j+1,min(len(string),j+4)):
                currentIpAddress[2]=string[j:k]
                currentIpAddress[3]=string[k:]
                if isValid(currentIpAddress[2]) and isValid(currentIpAddress[3]):
                    ipAddressFound.append('.'.join(currentIpAddress))
    return ipAddressFound

def isValid(string):
    stringAsInt=int(string)
    if stringAsInt<0 or stringAsInt>255:
        return False
    return len(str(stringAsInt))==len(string)

validIPAdresses(string)



# Underscorify Substring (Difficult)

string="testthis is a testtest to see if testestest it works"
substring='test'

# Space: O(N) and Time: O(N)

def underScorify(string,substring):
    locations=updateLocation(getLocations(string,substring))
    return puttingUnderScores(locations,string)

def getLocations(string,substring):
    locations=[]
    startIdx=0
    while startIdx<len(string) :
        nextIdx=string.find(substring,startIdx)
        if nextIdx!=-1:
            locations.append([nextIdx,nextIdx+len(substring)])
            startIdx=nextIdx+1
        else:
            break
    return locations 

def updateLocation(locations):
    if locations ==-1:
        return locations
    newLocation=[locations[0]]
    previous=newLocation[0]
    for i in range(1,len(locations)):
        current=locations[i]
        if current[0]<=previous[1]:
            previous[1]=current[1]
        else:
            newLocation.append(current)
            previous=current
    return newLocation
            
def puttingUnderScores(locations,string):
    locationIdx=0
    stringIdx=0
    finalChar=[]
    inBetweenChar=False
    i=0
    while stringIdx<len(string) and locationIdx<len(locations):
        if stringIdx==locations[locationIdx][i]:
            finalChar.append('_')
            inBetweenChar=not inBetweenChar
            if not inBetweenChar:
                locationIdx+=1
            i=0 if i==1 else 1
        finalChar.append(string[stringIdx])
        stringIdx+=1
    if locationIdx<len(locations):
        finalChar.append('_')
    if stringIdx<len(string):
        finalChar.append(string[stringIdx:])
    return ''.join(finalChar)

string="test this is a test to see if it works"
substring='test'
underScorify(string,substring)







# Generate Document (Easy)

characters="Bste!hetsi ogEAxpelrt x "
document="AlgoExpert is the Best!"

# Optimized Space: O(C) Time: O(N+M)   -----where C is unique characters nad N,M is length of characters and document
def generateDocument(characters,document):
    seenCount={}
    for character in characters:
        if character not in seenCount:
            seenCount[character]=0
        seenCount[character]+=1
    for character in document:
        if character not in seenCount or seenCount[character]==0:
            return False
        seenCount[character]-=1
    return True

generateDocument(characters,document)





# Reverse Word String (Medium)

string="AlgoExpert is the best!"

def reverseWordString(string):
    word=[]
    startOfWord=0
    for i in range(0,len(string)):
        if string[i]==' ':
            word.append(string[startOfWord:i])
            startOfWord=i
        elif string[startOfWord]==' ':
            word.append(' ')
            startOfWord=i
    word.append(string[startOfWord:])
    reverse(word)
    return ''.join(word)


def reverse(char):
    startIdx,endIdx=0,len(char)-1
    while startIdx<endIdx:
        char[startIdx],char[endIdx]=char[endIdx],char[startIdx]
        startIdx+=1
        endIdx-=1
    

reverseWordString(string)



# Pattern Matcher (Difficult)

pattern="xxyxxy"
string="gogopowerrangergogopowerranger"

# time: O(N^2+M), Space: O(N+M)
def patternMatcher(pattern, string):
    if len(string)<len(pattern):
        return []
    newPattern=getNewPattern(pattern)
    didSwitch=newPattern[0]!=pattern[0]
    counts={"x":0,"y":0}
    firstYPosition=getCountsAndFirstYPosition(newPattern,counts)
    if counts["y"]!=0:
        for XLen in range(1,len(string)):
            YLen=(len(string)-(counts["x"]*XLen))/counts["y"]
            if YLen<=0 or YLen%1!=0:
                continue
            else:
                YLen=int(YLen)
                yIdx=firstYPosition*XLen
                x=string[:XLen]
                y=string[yIdx:yIdx+YLen]
                potentialMatcher=map(lambda char: x if char=="x" else y,newPattern)
                if string=="".join(potentialMatcher):
                    return [x,y] if not didSwitch else [y,x]
    else:
        XLen=len(string)/counts["x"]
        if XLen%1==0:
            XLen=int(XLen)
            x=string[:XLen]
            potentialMatcher=map(lambda char:x , newPattern)
            if string=="".join(potentialMatcher):
                return [x,""] if not didSwitch else ["",x]
    return []



def getNewPattern(pattern):
    patternLetters=list(pattern)
    if patternLetters[0]=="x":
        return patternLetters
    else:
        return list(map(lambda char:"x" if char=="y" else "y",patternLetters))
    
def getCountsAndFirstYPosition(pattern,counts):
    firstYPosition=None
    for i,char in enumerate(pattern):
        counts[char]+=1
        if char=="y" and firstYPosition is None:
            firstYPosition=i
    return firstYPosition

patternMatcher(pattern, string)

# Leetcode Sum
# Similar so solved
def wordPattern(pattern, s):
    li = s.split(' ')
    di = {}
    if len(li) != len(pattern):
        return False

    for i, val in enumerate(pattern):
        if val in di and di[val] != li[i]:
            return False
        elif val not in di and li[i] in di.values():
            return False
        elif val not in di:
            di[val] = li[i]

    return di

pattern1 = "abba"
s1 = "dog cat cat dog"

wordPattern(pattern1, s1)



# Smallest Substring Containing (Very Difficult)

bigString="abcd$ef$axb$c$"
smallString="$$abf"

def smallestSubstringContaining(bigString,smallString):
    targetCharCounts=getCountSmallString(smallString)
    print(targetCharCounts)
    getSubStringBounds=getStringBounds(bigString,targetCharCounts)
    print(getSubStringBounds)
    return getSubString(bigString,getSubStringBounds)


def getCountSmallString(string):
    countSmallString={}
    for char in string:
        increasing(char,countSmallString)
    return countSmallString


def getStringBounds(bigString,targetCounts):
    subStringBounds=[0,float("inf")]
    countSubstring={}
    uniqueCharsString=len(targetCounts.keys())
    uniqueCharsDone=0
    leftIdx=0
    rightIdx=0
    
    while rightIdx<len(bigString):
        rightChar=bigString[rightIdx]
        if rightChar not in targetCounts:
            rightIdx+=1
            continue
        increasing(rightChar,countSubstring)
        if countSubstring[rightChar]==targetCounts[rightChar]:
            uniqueCharsDone+=1
        
        while uniqueCharsDone==uniqueCharsString and leftIdx<=rightIdx:
            subStringBounds=getCloserBounds(leftIdx,rightIdx,subStringBounds[0],subStringBounds[1])
            leftChar=bigString[leftIdx]
            if leftChar not in countSubstring:
                leftIdx+=1
                continue
            if countSubstring[leftChar]==targetCounts[leftChar]:
                uniqueCharsDone-=1
            decreasing(leftChar,countSubstring)
            leftIdx+=1
        rightIdx+=1
    return subStringBounds


def getCloserBounds(idx1,idx2,idx3,idx4):
    return [idx1,idx2] if idx2-idx1 < idx4-idx3 else [idx3,idx4]

def getSubString(string,stringBound):
    start,end=stringBound
    if end==float("inf"):
        return ""
    return string[start:end+1]


def increasing(char,countSmallString):
    if char not in countSmallString:
        countSmallString[char]=0
    countSmallString[char]+=1
    
def decreasing(char,countSmallString):
    countSmallString[char]-=1

smallestSubstringContaining(bigString,smallString)







# Longest Valid Parentheses

string="(()))("

# Brute force approach
# Leetcode name (Longest valid paranthesis) 
# Time: O(N^3), Space:O(N)
def longestValidParenthesis(string):
    maxLength=0
    
    for i in range(len(string)):
        for j in range(2,len(string)+1,2):
            if isBalanced(string[i:j]):
                currentLength=j-i
                maxLength=max(currentLength,maxLength)
    return maxLength

def isBalanced(string):
    openParensStack=[]
    
    for char in string:
        if char=="(":
            openParensStack.append(char)
        elif len(openParensStack)>0:
            openParensStack.pop()
        else:
            return False
    return len(openParensStack)==0

longestValidParenthesis(string)

string="(()))((()))("

# Optimised Approach
def longestValidParenthesis_optimised(string):
    maxLength=0
    opening=0
    closing=0
    
    for char in string:
        if char=="(":
            opening+=1
        else:
            closing+=1
            
        if opening==closing:
            maxLength=max(maxLength,opening+closing)
        elif closing>opening:
            opening=0
            closing=0
        
     
    opening=0
    closing=0
    for i in reversed(range(len(string))):
        char=string[i]
        
        if char=="(":
            opening+=1
        else:
            closing+=1
            
        if opening==closing:
            maxLength=max(maxLength,opening+closing)
        elif opening>closing:
            opening=0
            closing=0
            
    return maxLength
             

longestValidParenthesis_optimised(string)

a



# First Non-Repeating Character

string="abcdcaf"

def firstNonRepeatingCharacter(string):
    charTable=getCharAndCount(string)
    return firstCharacter(charTable,string)
    
def getCharAndCount(string):
    seenCharCount={}
    for i in range(len(string)):
        char=string[i]
        if char not in seenCharCount:
            seenCharCount[char]=0
        seenCharCount[char]+=1
    return seenCharCount

def firstCharacter(charTable,string):
    for i in range(len(string)):
        char=string[i]
        if charTable[char]==1:
            return i
    return -1

firstNonRepeatingCharacter(string)

s="abcab"

# Leetcode name (First Unique Character in a String)
#  Time: O(N) , Space: O(1)

def firstUniqChar(s):
    charTable=getCharAndCount(s)
    print(charTable)
    return firstCharacter(charTable,s)

def getCharAndCount(s):
    seenCharCount={}
    for i in range(len(s)):
        char=s[i]
        if char not in seenCharCount:
            seenCharCount[char]=0
        seenCharCount[char]+=1
    return seenCharCount

def firstCharacter(charTable,s):
    for i in range(len(s)):
        char=s[i]
        if charTable[char]==1:
            return i
    return -1

firstUniqChar(s)



# Semordnilap

# Time: O(n*m) , Space: O(n*m)
def semordnilap(words):
    wordsSet=set(words)
    dic=[]
    for word in words:
        reverse=word[::-1]

        if reverse in wordsSet and reverse!=word:
            dic.append([word,reverse])
            wordsSet.remove(word)
            wordsSet.remove(reverse)
    return dic

words=["dog", "hello", "god"]

semordnilap(words)

# Minimum Characters For Words

words=["this", "that", "did", "deed", "them!", "a"]

# Flow:
# Motha Hashtable
# chota hashtable for each word
# Go through both hashtable and update maximum
# Motha hashtable ch list madhe convert karra


# Time: O(n*L)  Space: O(C) ---where n is number of words, l is length of the longest word,
#  and c is the number of unique characters across all words
def minCharactersForWords(words):
    maxFrequency={}
    
    for word in words:
        charCount=getCharCountForWord(word)
        updateMaxFrequency(charCount,maxFrequency)
        
    return getArrayOfChar(maxFrequency)


def getCharCountForWord(string):
    charCountForWord={}
    for char in string:
        if char not in charCountForWord:
            charCountForWord[char]=0
        charCountForWord[char]+=1
    return charCountForWord

def updateMaxFrequency(charCount,maxFrequency):
    for char in charCount:
        frequency=charCount[char]
        if char not in maxFrequency:
            maxFrequency[char]=frequency
        else:
            maxFrequency[char]=max(frequency,maxFrequency[char])

def getArrayOfChar(maxFrequency):
    array=[]
    for char in maxFrequency:
        frequency=maxFrequency[char]
        
        for _ in range (frequency):
            array.append(char)
    return array
    
        

minCharactersForWords(words)

string='abccddcba'

def palindrome(string):
    leftIdx=0
    rightIdx=len(string)-1
    
    while leftIdx<rightIdx:
        if string[leftIdx]==string[rightIdx]:
            leftIdx+=1
            rightIdx-=1
        else:
            return False
    return True

palindrome(string)



# Practice: Again

string='xyz'
key=2

def caesarCipherEncryptor(string, key):
    key=2%26
    output=[]
    for letter in string:
        newLetters=getNewAlphabet(letter,key)
        output.append(newLetters)
    return ''.join(output)



def getNewAlphabet(letter,key):
    convertedNum=ord(letter) + key
    return chr(convertedNum) if convertedNum<=122 else chr(96+convertedNum%122)

caesarCipherEncryptor(string, key)

365%122



string="AAAAAAAAAAAAABBCCCCDD"

def runLengthEncoding(string):
    output=[]
    currentLength=1
    for i in range(1,len(string)):
        currentWord=string[i]
        previousWord=string[i-1]
        if currentWord!=previousWord or currentLength==9:
            output.append(str(currentLength))
            output.append(previousWord)
            currentLength=0
        currentLength+=1 
    output.append(str(currentLength))
    output.append(string[(len(string)-1)])
    return ''.join(output)

runLengthEncoding(string)



# Generate Document

characters="Bste!hetsi ogEAxpelrt x "
document="AlgoExpert is the Best!"

def generateDocument(characters, document):
    seenChar={}
    
    for char in characters:
        if char not in seenChar:
            seenChar[char]=0
        seenChar[char]+=1
    print(seenChar)
    
    for char in document:
        if char not in seenChar or seenChar[char]==0:
            return False
        seenChar[char]-=1
    return True

generateDocument(characters, document)



string="abaxyzzyxf"

def longestPalindromicSubstring(string):
    currentlongest=[0,1]
    for i in range(1,len(string)):
        odd=getLongestPalindrome(string,i-1,i+1)
        even=getLongestPalindrome(string,i-1,i)
        longest=max(odd,even,key=lambda x:x[1]-x[0])
        print(longest)
        currentlongest=max(currentlongest,longest,key=lambda x:x[1]-x[0])
        
    return string[currentlongest[0]:currentlongest[1]]

def getLongestPalindrome(string,leftIdx,rightIdx):
    while leftIdx>=0 and rightIdx<len(string):
        if string[leftIdx]!=string[rightIdx]:
            break
        leftIdx-=1
        rightIdx+=1
    return [leftIdx+1,rightIdx]

longestPalindromicSubstring(string)

string=["yo", "act", "flop", "tac", "foo", "cat", "oy", "olfp"]

def groupAnagrams(words):
    anagrams={}
    
    for word in string:
        newWord=''.join(sorted(word))
        if newWord not in anagrams:
            anagrams[newWord]=[word]
        else:
            anagrams[newWord].append(word)
    return list(anagrams.values())

groupAnagrams(words)



a='cda'

sorted(a)




string="1921680"

def validIPAddresses(string):
    ipAddressFound=[]
    
    for i in range(1,min(len(string),4)):
        print(len(string))
        currentIpAddress=["","","",""]
        currentIpAddress[0]=string[:i]
        if not isValid(currentIpAddress[0]):
            continue
        for j in range(i+1,min(len(string),i+4)):
            print(len(string))
            currentIpAddress[1]=string[i:j]
            if not isValid(currentIpAddress[1]):
                continue
            for k in range(j+1,min(len(string),j+4)):
                print(len(string))
                currentIpAddress[2]=string[j:k]
                currentIpAddress[3]=string[k:]
                if isValid(currentIpAddress[2]) and isValid(currentIpAddress[3]):
                    ipAddressFound.append('.'.join(currentIpAddress))
                    
    return ipAddressFound

def isValid(string):
    num=int(string)
    if num<0 or num>255:
        return False
    return len(string)==len(str(num))
            

validIPAddresses(string)




---------------------------------------------------------

string="AlgoExpert is the best!"

def reverseWordsInString(string):
    word=[]
    startOfWord=0
    for i in range(0,len(string)):
        if string[i]==' ':
            word.append(string[startOfWord:i])
            startOfWord=i
        elif string[startOfWord]==' ':
            word.append(' ')
            startOfWord=i
    word.append(string[startOfWord:])
    reverse(word)
    return ''.join(word)

def reverse(char):
    startIdx,endIdx=0,len(char)-1
    while startIdx<endIdx:
        char[startIdx],char[endIdx]=char[endIdx],char[startIdx]
        startIdx+=1
        endIdx-=1

reverseWordsInString(string)










--------------------------------------------------------------------

string="testthis is a testtest to see if testestest it works"
substring="test"

def underscorifySubstring(string, substring):
    newLocations=updateLocation(getLocations(string,substring))
    return newLocations

def getLocations(string,substring):
    location=[]
    startIdx=0
    while startIdx<len(string):
        newIdx=string.find(substring,startIdx)
        if newIdx!=-1:
            location.append([newIdx,newIdx+len(substring)+1])
            startIdx=newIdx+1
        else:
            break
    return location



def updateLocation(location):
    if not len(location):
        return location
    newLocation=[location[0]]
    previous=newLocation[0]
    for i in range(1,len(location)):
        current=location[i]
        if current[0]<=previous[1]:
            previous[1]=current[1]
        else:
            newLocation.append(current)
            previous=current
    return newLocation

updateLocation(location)

def merge(locations):
        output=[locations[0]]
        for start,end in locations[1:]:
            lastEnd=output[-1][1]
            if lastEnd>=start:
                output[-1][1]=max(lastEnd,end)
            else:
                output.append([start,end])        
        return output   

   
location=[[0, 5], [14, 23], [18, 23], [33, 44], [36, 41], [39, 45]]




