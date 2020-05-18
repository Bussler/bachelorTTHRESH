#include "HuffmanCode.h"



HuffmanCode::HuffmanCode()
{
}


HuffmanCode::~HuffmanCode()
{
}

//Method to encode and write Rle-data with Huffman-Coding
void HuffmanCode::encodeData(std::vector<std::vector<int>>& rleVek)
{
	//create freq model
	std::map<int, int> freq;// key -> (count of key)
	for (int i = 0; i < rleVek.size(); i++) {
		for (int j = 0; j < rleVek[i].size(); j++) {
			freq[rleVek[i][j]] += 1; //count the occurences of the key
		}

	}

	//safe freq model
	uint64_t freqSize = freq.size();
	BitIO::writeBit(freqSize, 64);

	for (auto it = freq.begin(); it != freq.end(); it++) {
		uint64_t key = it->first;
		uint64_t prob = it->second;

		//encode key len, then key to safe space
		uint8_t keyLen = 0;
		uint64_t keyCopy = key;
		while (keyCopy > 0) {
			keyCopy >>= 1;
			keyLen++;
		}
		if (keyLen == 0)
			keyLen = 1;

		BitIO::writeBit(keyLen, 6);
		BitIO::writeBit(key, keyLen);//save key 

		uint8_t probLen = 0;
		uint64_t probCopy = prob;
		while (probCopy > 0) {
			probCopy >>= 1;
			probLen++;
		}
		if (probLen == 0)
			probLen = 1;

		BitIO::writeBit(probLen, 6);
		BitIO::writeBit(prob, probLen);//save prob
	}

	//safe rleVekSizes
	uint64_t rleVekSize = rleVek.size();
	BitIO::writeBit(rleVekSize, 64); //save the number of elements to write

	for (int i = 0;i < rleVekSize;i++) {
		uint64_t rleSize = rleVek[i].size();
		BitIO::writeBit(rleSize, 64);//save the number of symbols to encode
	}

	//build huffman tree
	std::priority_queue<TreeNode*, std::vector<TreeNode*>, myComparator > q;
	std::map<int, TreeNode*> elements;//vector to hold all alements for fast access

	createHuffmanTree(q, freq, elements);

	std::stack<bool> path;//path of goalNode from root to node

	//encode data with huffman coding
	for (int itOverRle = 0; itOverRle < rleVek.size(); itOverRle++) {
		for (int i = 0;i < rleVek[itOverRle].size();i++) {
			TreeNode * cur = elements[rleVek[itOverRle][i]];//get current symbol

			while (cur->nextNode != nullptr) {//find path of symbol to root
				path.push(cur->code);
				cur = cur->nextNode;
			}

			while (!path.empty()) {//encode symbol with path
				BitIO::writeBit(path.top(), 1);
				path.pop();
			}
		}
	}

}

//reads in and decodes Huffman-Code Data and stores in vector for later calculation of core
void HuffmanCode::decodeData(std::vector<std::vector<int>>& rleVek)
{
	//read and recreate the saved frequenzy table
	uint64_t freqSize = BitIO::readBit(64); //table size safed with 64 bit

	std::map<int, int> freq;//key -> frequenzy

	for (int i = 0;i < freqSize;i++) {
		uint64_t keyLen = BitIO::readBit(6);
		uint64_t key = BitIO::readBit(keyLen);
		uint64_t probLen = BitIO::readBit(6);
		uint64_t prob = BitIO::readBit(probLen);

		freq[key] = prob;
	}

	//read in safed data for vector
	uint64_t rleVekSize = BitIO::readBit(64);
	std::vector<uint64_t> rleSizes;
	for (int i = 0;i < rleVekSize;i++) {
		uint64_t curVekSize = BitIO::readBit(64);
		rleSizes.push_back(curVekSize);
	}

	
	//create huffman tree
	std::priority_queue<TreeNode*, std::vector<TreeNode*>, myComparator > q;
	std::map<int, TreeNode*> elements;//vector to hold all alements

	createHuffmanTree(q, freq, elements);

	std::stack<bool> path;//path from root to node

	//decode data with huffman coding
	for (int i = 0;i < rleVekSize;i++) {
		std::vector<int> curPlane;
		for (int j = 0;j < rleSizes[i];j++) {
			//decode symbol
			TreeNode * cur = q.top();
			while (!cur->isLeaf) {//read in bits until we find a leaf-node(encoded symbol)
				bool curBit = BitIO::readBit(1) ? 1 : 0;//read in bit from data
				if (curBit)//1 right
					cur = cur->right;
				else {//0 left
					cur = cur->left;
				}
			}

			curPlane.push_back(cur->data);//push back found symbol
		}
		rleVek.push_back(curPlane);
	}

}

//helper Method to create symbol/ leaf-nodes
HuffmanCode::TreeNode * HuffmanCode::createNodeLeaf(int data, int freq)
{
	TreeNode* nNode = (TreeNode*) malloc(sizeof(TreeNode));
	nNode->data = data;
	nNode->freq = freq;

	nNode->nextNode = nullptr;
	nNode->isLeaf = true;

	return nNode;
}

//helper Method to create inner Nodes in Huffman Tree
HuffmanCode::TreeNode * HuffmanCode::createInnerNode(int freq, TreeNode * left, TreeNode * right)
{
	TreeNode* nNode = (TreeNode*)malloc(sizeof(TreeNode));
	nNode->freq = freq;
	nNode->left = left;
	nNode->right = right;

	nNode->nextNode = nullptr;
	nNode->isLeaf = false;

	return nNode;
}

//creates Huffman Tree from freq-Tables in queue q
void HuffmanCode::createHuffmanTree(std::priority_queue<TreeNode*, std::vector<TreeNode*>, myComparator>& q, std::map<int, int>& freq, std::map<int, TreeNode*>& elements)
{
	//push in nodes from freq-Tables: Symbols
	for (auto it = freq.begin(); it != freq.end(); it++) {
		TreeNode* cur = createNodeLeaf(it->first, it->second);
		q.push(cur);
		elements[it->first] = cur;
	}

	//merge nodes to create Huffman-Tree in q
	while (q.size() > 1) {
		TreeNode* cur1 = q.top();
		q.pop();

		TreeNode* cur2 = q.top();
		q.pop();

		TreeNode* nNode = createInnerNode(cur1->freq + cur2->freq, cur1, cur2);//create parent node with joint frequency

		cur1->nextNode = nNode; //left: 0
		cur1->code = 0;

		cur2->nextNode = nNode; //right: 1
		cur2->code = 1;

		q.push(nNode);
	}
}

