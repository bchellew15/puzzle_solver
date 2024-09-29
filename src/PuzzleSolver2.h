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
	Mat edgeImg;
	vector<Mat> rotEdgeImgs;
	vector<double> rotEdgeImgDegrees;
	static int edgeImgBuffer;  // 10 looked visually good
	Point rasterShift;  // amount edge center is shifted when made into raster image
	bool isEdgeVar = false;
	double thetaCorrection;  // for flat edges
	int shiftCorrection; // for flat edges

	bool isEdge();
	void rotate_edge(vector<Point> unrotated, double theta);
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
	int number; //the number of the piece
	double theta = 0; // counterclockwise rotation required for the image of this piece. unused?
	int rightIndex; //index of edge pointing towards rightNeighbor.
	bool isConnected = false; // whether this piece has been connected to the puzzle
	static double scalingLength;
	static double avgBrightness;
	Point correctionShift;  // remove this
	Point correctionShiftLeft;
	Point correctionShiftUp;
	double correctionTheta;  // remove this
	double correctionThetaLeft;
	double correctionThetaUp;
	double finalRotationCorrection;
	static double edgeShrinkFactor; // shrink edges for faster processing

	void process(bool verbose=false); // process the image and get edge shapes
	bool isCorner();
	int countEdges(); //returns number of edges on the piece. 2 for corner, 1 for edge
	//bool isEdge();  //returns true for corners
	int orientRoot();
	static int nextIndex(int index); //returns next index in clockwise order
	static int prevIndex(int index);
	static int oppIndex(int index);
	int downIndex();
	int upIndex();
	int leftIndex();
	void print();
	double rotationAngle();
	double width();
	double height();
	Point center();  // unused?
	vector<Point> constructEdge(vector<Point> outline, int firstIdx, int secondIdx);
	void scale(double factor);
	void shift(Point s, Size newSize);
	void rotate(Point rotationCenter, double theta);

	//constructors:
	//PuzzlePiece();
};

class PieceMatch {
public:
	PuzzlePiece *piece;
	int edgeIndex;
	double theta;
	double thetaLeft;
	double thetaUp;
	Point shift;
	Point shiftLeft;
	Point shiftUp;
	double score;
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
	static PieceMatch matchEdges(EdgeOfPiece firstEdge, EdgeOfPiece other, bool verbose=false);
	PieceMatch match(PuzzlePiece *piece, int edgeIndex, bool verbose=false); //finds the matching piece
	PieceMatch match2(PuzzlePiece *leftPiece, int edgeIndexOfLeft, PuzzlePiece *upPiece, int edgeIndexOfUp, bool verbose);

	void process(bool verbose=false);
	void assemble(bool verbose=false);
	void print();
	void display(bool verbose=false, bool checkRotation=true);
};

double edgeComparisonScore(Mat edge1, Mat edge2);

#endif /* PUZZLESOLVER2_H_ */
