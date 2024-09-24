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
	double match(EdgeOfPiece other, bool verbose=false);
};

class PuzzlePiece {
public:
	PuzzlePiece();
	PuzzlePiece(Mat m, int i);

	Mat img;
	vector<EdgeOfPiece> edges = vector<EdgeOfPiece>(4);  //edges in order: top, right, bottom, left
	vector<Point> outline;
	Rect core;
	int number; //the number of the piece
	double theta = 0; // counterclockwise rotation required for the image of this piece. unused?
	int rightIndex; //index of edge pointing towards rightNeighbor.
	bool isConnected = false; // whether this piece has been connected to the puzzle
	static double scalingLength;

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
	double rotationAngle();
	double width();
	double height();
	Point center();
	vector<Point> constructEdge(vector<Point> outline, int firstIdx, int secondIdx);
	void scale(double factor);
	void shift(Point s, Size newSize);
	void rotate(Point rotationCenter, double theta);

	//search through all the pieces until it finds a match
	//returns a pointer the piece with matching index
	//if no match, returns NULL
	pair<PuzzlePiece*, int> match(int edgeIndex, PuzzlePiece pieces[], int numPieces, bool verbose=false); //finds the matching piece

	//constructors:
	//PuzzlePiece();
};

class Puzzle {
public:

	Puzzle(int _numPieces, PuzzlePiece _pieces[]);

	PuzzlePiece *pieces;
	int numPieces;
	vector<vector<PuzzlePiece*>> completedPuzzle;
	int rows;
	int columns;
	vector<double> rowHs;
	vector<double> colWs;

	void process(bool verbose=false);
	void assemble(bool verbose=false);
	void print();
	void display(bool verbose=false, bool checkRotation=true);
};

double edgeComparisonScore(vector<Point> edge1, vector<Point> edge2);

#endif /* PUZZLESOLVER2_H_ */
