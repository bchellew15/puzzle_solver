/*
 * PuzzleSolver2.h
 *
 *  Created on: Aug 23, 2024
 *      Author: blakechellew
 */

#ifndef PUZZLESOLVER2_H_
#define PUZZLESOLVER2_H_

class EdgeOfPiece {
public:
	vector<Point> edge;
	bool isEdgeVar = false;

	bool isEdge();
	void rotate_edge(vector<Point> unrotated, double theta);
	double match(EdgeOfPiece other);
};

class PuzzlePiece {
public:
	PuzzlePiece();
	PuzzlePiece(Mat m, int i, bool verbose=true);

	Mat img;
	vector<EdgeOfPiece> edges = vector<EdgeOfPiece>(4);  //edges in order: top, right, bottom, left
	PuzzlePiece *leftNeighbor;
	PuzzlePiece *rightNeighbor;
	PuzzlePiece *upNeighbor;
	PuzzlePiece *downNeighbor;
	Rect core;
	int number; //the number of the piece
	double theta = 0; // counterclockwise rotation required for the image of this piece
	int rightIndex; //index of edge pointing towards rightNeighbor.
	bool isConnected = false; // whether this piece has been connected to the puzzle

	void process(bool verbose=false); // process the image and get edge shapes
	bool isCorner();
	int countEdges(); //returns number of edges on the piece. 2 for corner, 1 for edge
	//bool isEdge();  //returns true for corners
	int orientRoot();
	static int nextIndex(int index); //returns next index in clockwise order
	static int prevIndex(int index);
	static int oppIndex(int index);
	int downIndex();
	void print();
	void rotate(); //necessary? how did I code it before?
	vector<Point> constructEdge(vector<Point> outline, int firstIdx, int secondIdx);

	//search through all the pieces until it finds a match
	//returns a pointer the piece with matching index
	//if no match, returns NULL
	pair<PuzzlePiece*, int> match(int edgeIndex, PuzzlePiece pieceArray[], int pieceArraySize); //finds the matching piece

	//constructors:
	//PuzzlePiece();
};

#endif /* PUZZLESOLVER2_H_ */
