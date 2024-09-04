#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> //for setw(n)
#include "PuzzleSolver2.h"
using namespace std;

//try printing out piece 1 and its right neighbor.
//matching piece: will it help to return pointer as a reference param?
//passing an array as a call-by reference?
//match isn't returning correctly

//takes numbered images of puzzle pieces and prints out a grid showing where the pieces should go
//pieces are labeled edge, corner, or middle

//make more functions to break it up
//protect against errors
//check for small puzzle sizes
//how is it accessing edges? that's private.
//passing arrays of pointers?
//range-based for loop is c++ 11 only?
//a lot of the functions should not be part of any class.
//protect against infinite loops

// MISC:
// match() function should also return the index of the matching edge (finds it anyway and then I have to call matchingEdge after)
//    (but can you return a tuple in C++?)

int main() {

	//load the pieces using a dynamic array:
	string sizeStr;
	cout << "How many pieces?" << endl;
	cin >> sizeStr;
	int size = stoi(sizeStr);
	cout << size << " pieces" << endl;

	// one of the * is bc array of pointers. the other is bc creating dynamic array.
	// but it didn't work!
	// PuzzlePiece **pieces = new PuzzlePiece*[size];

	// previous code to create array of pointers then initialize each element:
	PuzzlePiece *pieces[size]; //create an array of pointers
	for(int i = 0; i < size; i++) {
		pieces[i] = new PuzzlePiece;
	}

	//take input:
	for(int i = 0; i < size; i++) {
			cout << "Piece " << i+1 << ":" << endl;
			pieces[i]->take_data();
			pieces[i]->number = i+1;
	}

	cout<<"Loaded!" << endl;

	//find a corner:
	int firstCornerIndex = 0;
	while(firstCornerIndex < size && !((pieces[firstCornerIndex])->isCorner())) {
		firstCornerIndex++;
	}
	//firstCornerIndex is now the index of a corner

	//cout << "first corner is " << firstCornerIndex << endl;

	//set downIndex and rightIndex:
	pieces[firstCornerIndex]->rightIndex = pieces[firstCornerIndex]->firstConnection();
	pieces[firstCornerIndex]->downIndex = PuzzlePiece::nextIndex(pieces[firstCornerIndex]->rightIndex);

	//cout << "right index is " << pieces[firstCornerIndex]->rightIndex << endl;
	//cout << "down index is " << pieces[firstCornerIndex]->downIndex << endl;

	PuzzlePiece *cursor;

	cout << "starting matching" << endl;

	//loop the cursor through the other rows:
	cursor = pieces[firstCornerIndex];
	while(cursor != NULL) {
		//look for a matching piece and set its orientation:

		//cout << "looping with piece " << cursor->number << endl;

		//find the matching piece:
		PuzzlePiece *matchingPiece;
		matchingPiece = cursor->match(cursor->downIndex,pieces,size);

		//link them:
		cursor->downNeighbor = matchingPiece;

		//set up the down neighbor if not null:
		if(cursor->downNeighbor != NULL) {
			//find the index of the matching edge, and assign to the edge
			int edgeIndex = matchingPiece->matchingEdgeIndex(cursor->edges[cursor->downIndex].id_string);

			//cout << "down neighbor: " << matchingPiece->number << endl;
			//orient the piece (set rightIndex and downIndex:)
			matchingPiece->rightIndex = PuzzlePiece::nextIndex(edgeIndex);
			matchingPiece->downIndex = PuzzlePiece::oppIndex(edgeIndex);
		}

		//start another loop that goes horizontal, with a subCursor
		PuzzlePiece *subCursor;
		subCursor = cursor;
		while(subCursor != NULL) {

			//cout << "sublooping with piece " << subCursor->number << endl;

			//find the matching piece:
			PuzzlePiece *subMatchingPiece;
			subMatchingPiece = subCursor->match(subCursor->rightIndex,pieces,size);

			//link them:
			subCursor->rightNeighbor = subMatchingPiece;

			//if(subCursor->number == 1) {
				//cout << "subcursor right neighbor is " << s
			//}

			if(subCursor->rightNeighbor != NULL) {
				//cout << "right neighbor: " << subMatchingPiece->number << endl;

				//find the index of the matching edge, and assign to the edge
				int subEdgeIndex = subMatchingPiece->matchingEdgeIndex(subCursor->edges[subCursor->rightIndex].id_string);

				//orient the piece (set rightIndex)
				subMatchingPiece->rightIndex = PuzzlePiece::oppIndex(subEdgeIndex);
			}

			//still works if NULL
			//cout << "done sublooping with piece " << subCursor->number << endl;
			subCursor = subCursor->rightNeighbor;

		}

		//still works if NULL
		//cout << "done looping" << endl;
		cursor = cursor->downNeighbor;
	}

	cout << "done matching" << endl;

	/*if(pieces[firstCornerIndex]->downNeighbor == NULL)
		cout << "DOWN IS NULL";
	else
		cout << "DOWN IS WORKING";

	if(pieces[firstCornerIndex]->rightNeighbor == NULL)
			cout << "RIGHT IS NULL";
		else
			cout << "RIGHT IS WORKING";
	*/

	//print:
	cursor = pieces[firstCornerIndex];
	while(cursor != NULL) {

		//start another loop that goes horizontal, with a subCursor
		PuzzlePiece *subCursor;
		subCursor = cursor;
		while(subCursor != NULL) {
			subCursor->print();

			//still works if NULL
			subCursor = subCursor->rightNeighbor;
		}
		cout << endl;

		//still works if NULL
		cursor = cursor->downNeighbor;
	}

	return 0;
}

Point::Point(int xArg, int yArg) {
	x = xArg;
	y = yArg;
}

Point::Point() {
	x = 0;
	y = 0;
}

void Point::print() {
	cout << "(" << x << ", " << y << ")";
}

void PuzzlePiece::take_data() {
	cout << "Enter the 4 strings, in clockwise order, pressing enter after each one.\n"
			<< "Type x for any outside edges."<< endl;
	string var1; string var2; string var3; string var4;
	cin >> var1 >> var2 >> var3 >> var4;

	edges[0].setId(var1);
	edges[1].setId(var2);
	edges[2].setId(var3);
	edges[3].setId(var4);
}

void EdgeOfPiece::setId(string str) {
	id_string = str;
}

//returns true if 2 or less edges
bool PuzzlePiece::isCorner() {
	if(countEdges() >= 2) return true;
	else return false;
}

int PuzzlePiece::countEdges() {
	int count = 0;
	for(int i = 0; i < 4; i++) {
		if(edges[i].id_string == "x") count++;
	}
	return count;
}

int PuzzlePiece::firstConnection() {
	if(edges[0].id_string == "x") {
		if(edges[1].id_string == "x") {
			return 2;
		}
		else return 1;
	}
	else if(edges[1].id_string == "x") return 3;
	else return 0;
}

// could have done this with mod
int PuzzlePiece::oppIndex(int index) {
	return (index + 2) % 4;
}

int PuzzlePiece::nextIndex(int index) {
	return (index + 1) % 4;
}

int PuzzlePiece::matchingEdgeIndex(string s) {
	for(int i = 0; i < 4; i++) {
		if(edges[i].id_string == s) return i;
	}
	return -1; //no match found
}

PuzzlePiece* PuzzlePiece::match(int edgeIndex, PuzzlePiece *pieceArray[], int pieceArraySize) {
	//search through edges to find a match
	for(int k = 0; k < pieceArraySize; k++) {
		for(int i = 0; i < 4; i++) {
			//second condition is to make sure it's not the same piece. re-write in a more readable way.
			// (note: should just start this function by checking if "edgeIndex" refers to an edge.)
			if ( pieceArray[k]->edges[i].id_string != "x" && pieceArray[k]->edges[i].id_string == edges[edgeIndex].id_string && pieceArray[k]->number != number ) {
				PuzzlePiece *temp = pieceArray[k];
				// cout << pieceArray[k]->edges[i].id_string << " from piece " << pieceArray[k]->number << " matches " << edges[edgeIndex].id_string << " from piece " << number << endl;
				return temp;
			}
		}
	}
	//cout << "no match for piece " << number << endl;
	return NULL;
}

void PuzzlePiece::print() {
	cout << setw(4) << number;
}

//idea: use the maze searching algorithm to start with the first puzzle piece.
//it will save about one O(N) operation, finding the first corner.

//testing:
//I can test once I have everything but match() and loading in the pieces.
//draw a grid and mark the edges. Mix up the pieces. Number them. Enter the edge letters into the computer.

//deciding the dimensions
//there are "piece" objects, with 4 numbered edges

//Notes:
//distance between points should also be comparable
//a better algorithm would make all the connections for each piece,
//and also deal with cases where there are multiple possible matches.
//The program should be able to print some diagnostics to help with improvement
