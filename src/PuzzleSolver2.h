/*
 * PuzzleSolver2.h
 *
 *  Created on: Aug 23, 2024
 *      Author: blakechellew
 */

#ifndef PUZZLESOLVER2_H_
#define PUZZLESOLVER2_H_

class EdgeOfPiece {
	//an edge is a collection of points
	//should be standardized so that they start at (0,0) and end at (100,0)
//private:
	//Point *points;
	//static int NUM_POINTS; //each edge should have the same number of points

public:
	std::string id_string;
	void setId(std::string str);
	//a function to access the points for use in the PuzzlePiece comparison function
	//constructor: somehow take in a curve and turn it into points
};

class PuzzlePiece {
public:
	EdgeOfPiece edges[4];  //edges should be in clockwise order
	PuzzlePiece *leftNeighbor;
	PuzzlePiece *rightNeighbor;
	PuzzlePiece *upNeighbor;
	PuzzlePiece *downNeighbor;
	int number; //the number of the piece
	int rightIndex; //index of edge pointing towards rightNeighbor.
	int downIndex;
	bool isCorner();
	int countEdges(); //returns number of edges on the piece. 2 for corner, 1 for edge
	//bool isEdge();  //returns true for corners
	int firstConnection();
	static int nextIndex(int index); //returns next index in clockwise order
	static int oppIndex(int index);
	int matchingEdgeIndex(std::string s); //returns the index of the matching edge
	void print();

	//search through all the pieces until it finds a match
	//returns a pointer the piece with matching index
	//if no match, returns NULL
	PuzzlePiece* match(int edgeIndex, PuzzlePiece *pieceArray[], int pieceArraySize); //finds the matching piece

	//asks for 4 strings from the user. Enters these as edges' id_strings.
	void take_data();

	//constructors:
	//PuzzlePiece();
};

#endif /* PUZZLESOLVER2_H_ */
