#pragma once
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <map>
#include <stack>
#include "VolInputParser.h"


class HuffmanCode
{

public:

	struct TreeNode
	{
		int data;
		int freq;

		bool code;
		bool isLeaf = false;

		TreeNode * nextNode;
		TreeNode * left;
		TreeNode * right;

	};

	class myComparator
	{
	public:
		int operator() (const TreeNode * n1, const TreeNode * n2)
		{
			return n1->freq > n2->freq;
		}
	};


	HuffmanCode();
	~HuffmanCode();

	void encodeData(std::vector<std::vector<int>>& rleVek);
	void decodeData(std::vector<std::vector<int>>& rleVek);

	TreeNode* createNode(int data, int freq);
	TreeNode* createNode(int freq, TreeNode* left, TreeNode* right);

	void createHuffmanTree(std::priority_queue<TreeNode*, std::vector<TreeNode*>, myComparator >& q, std::map<uint64_t, uint64_t>& freq, std::map<uint64_t, TreeNode*>& elements);

};

