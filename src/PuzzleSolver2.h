/*
 * PuzzleSolver2.h
 *
 *  Created on: Aug 23, 2024
 *      Author: blakechellew
 */

#ifndef PUZZLESOLVER2_H_
#define PUZZLESOLVER2_H_

class EdgeMatch {
public:
	double theta;
	Point shift;
	double score;
};

class EdgeOfPiece {
public:
	vector<Point> edge;
	Mat edgeImg;
	vector<Mat> rotEdgeImgs;
	vector<double> rotEdgeImgAngles;
	Point rasterShift;  // amount edge center is shifted when made into raster image
	bool isEdge = false;
	double rotCorrection;  // for flat edges
	int shiftCorrection; // for flat edges
	static int edgeImgBuffer;  // 10 looked visually good
	static double edgeShrinkFactor; // shrink edges for faster processing
	static int pixelShift; // for edge comparison

	void processEdge();
	static double edgeComparisonScore(Mat edge1, Mat edge2);
	static EdgeMatch matchEdges(EdgeOfPiece firstEdge, EdgeOfPiece other, bool verbose=false);
};

class PuzzlePiece {
public:
	PuzzlePiece();
	PuzzlePiece(Mat m, int i);

	Mat img;
	vector<EdgeOfPiece> edges = vector<EdgeOfPiece>(4);  //edges in order: top, right, bottom, left
	vector<Point> outline;
	Rect core;
	vector<Point> midpoints = vector<Point>(4);
	int number;
	int rightIndex; //index of edge pointing towards rightNeighbor.
	double theta = 0; // counterclockwise rotation required for the image of this piece. unused?
	bool isConnected = false; // whether this piece has been connected to the puzzle
	bool isEdge;
	static double scalingLength;
	static double avgBrightness;
	Point correctionShiftLeft;
	Point correctionShiftUp;
	double correctionAngleLeft;
	double correctionAngleUp;
	double finalCorrectionAngle;  // factors in final rotation of neighboring pieces

	void process(bool verbose=false); // process the image and get edge shapes
	int scan(vector<Point> scanLine, Point increment, int scanDepth, int maxBuffer);
	bool isCorner();
	int countEdges(); //returns number of edges on the piece. 2 for corner, 1 for edge
	//bool isEdge();  //returns true for corners
	int rootRightIndex();
	static int nextIndex(int index); //returns next index in clockwise order
	static int prevIndex(int index);
	static int oppIndex(int index);
	int downIndex();
	int upIndex();
	int leftIndex();
	double rotationAngle();
	double width();
	double height();
	Point center();
	vector<Point> constructEdge(int firstIdx, int secondIdx);
	void scale(double factor);
	void shift(Point s);
	void rotate(Point rotationCenter, double theta);
	void cycleMidpoints();
};

class PieceMatch {
public:
	EdgeMatch match;
	PuzzlePiece *piece = nullptr;
	int edgeIndex;
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

	// search through all the pieces until it finds a match
	// returns a pointer the piece with matching index
	// if no match, returns NULL
	PieceMatch match(PuzzlePiece *piece, int edgeIndex, bool edgesOnly, bool verbose=false); //finds the matching piece
	vector<PieceMatch> match2(PuzzlePiece *leftPiece, int edgeIndexOfLeft, PuzzlePiece *upPiece, int edgeIndexOfUp, bool verbose);

	void process(bool verbose=false);
	void assemble(bool verbose=false);
	void print();
	void display(bool verbose=false);
};

Point rotatePoint(Point p, Mat t);

#endif /* PUZZLESOLVER2_H_ */
