#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> //for setw(n)
#include <cmath>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/shape.hpp>
using namespace cv;

#include "PuzzleSolver2.h"

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

	// get number of pieces.
	// todo: count the pieces in the folder
	string numPiecesStr;
	cout << "How many pieces?" << endl;
	cin >> numPiecesStr;
	int numPieces = stoi(numPiecesStr);
	cout << numPieces << " pieces" << endl;

	// load images
	// todo: check if the file loaded properly
	Mat images[numPieces];
	string dir = "/Users/blakechellew/Documents/Code/workspace/PuzzleSolver2/Image_Files/Piece";
	for( int i = 0; i < numPieces; i++) {
		string filename = dir + to_string(i+1) + ".jpeg";
		images[i] = imread(filename);
	}

	// try displaying a piece
	namedWindow("temp");
	imshow("temp", images[0]);
	waitKey(0);
	destroyWindow("temp");

	PuzzlePiece pieces[numPieces];
	for(int i = 0; i < numPieces; i++) {
		pieces[i] = PuzzlePiece(images[i], i, false); // last argument is "verbose"
	}

	// test: compare all edges to each other
	/*
	for(int i = 0; i < numPieces-1; i++) {
		for(int j = i+1; j < numPieces; j++) {
			for(int k = 0; k < 4; k++) {
				for(int l = 0; l < 4; l++) {
					if(pieces[i].edges[k].isEdgeVar || pieces[j].edges[l].isEdgeVar) continue;
					cout << pieces[i].edges[k].match(pieces[j].edges[l]);
					cout << " pieces (" << i+1 << ", " << j+1 << ") edges (" << k+1 << ", " << l+1 << ")" << endl;
				}
			}
		}
	}
	*/

	// edge test results:
	// avg dist between 2 and 4 pixels for edges.
	// avg dist > 100 for non-edges

	// test edge detection and counting
	/*
	for(int i = 0; i < numPieces; i++) {
		cout << "Piece " << i << endl;
		for(int j = 0; j < 4; j++) {
			cout << "Edge " << j+1 << ": " << pieces[i].edges[j].isEdgeVar << endl;
		}
		cout << "Number of edges: " << pieces[i].countEdges() << endl;
		cout << "Is corner? " << pieces[i].isCorner() << endl;
	}
	*/

	cout<<"Loaded!" << endl;

	// todo: make this an "assemble" function

	// find a corner
	int firstCornerIdx = 0;
	while(firstCornerIdx < numPieces && !(pieces[firstCornerIdx]).isCorner()) {
		firstCornerIdx++;
	}
	if(firstCornerIdx >= numPieces) {
		cout << "ERROR: no corners found" << endl;
		return 0;
	}

	cout << "First corner at index " << firstCornerIdx << endl;

	// test: display full puzzle
	// create a fake puzzle
	PuzzlePiece *root2 = &pieces[firstCornerIdx];
	root2->rightNeighbor = &pieces[3];
	root2->rightIndex = 0;
	pieces[3].rightIndex = 1;
	root2->downNeighbor = &pieces[2];
	pieces[2].rightNeighbor = &pieces[1];
	pieces[2].rightIndex = 2;
	pieces[1].rightIndex = 3;

	displayPuzzle(root2);

	return 0;

	PuzzlePiece *root = &pieces[firstCornerIdx];
	root->isConnected = true;
	// todo: check if pieces matched reaches numPieces
	int piecesMatched = 1;
	// set the orientation of the root piece
	root->rightIndex = root->orientRoot();

	cout << "Start matching" << endl;

	// wonder if there is a cleaner way to write these loops e.g. w less duplication

	PuzzlePiece *rowCursor = root;

	while(rowCursor != nullptr && piecesMatched < numPieces) {

		// iterate down to the bottom
		PuzzlePiece *columnCursor = rowCursor;
		while(columnCursor != nullptr && piecesMatched < numPieces) {

			cout << "Column cursor: piece " << columnCursor->number << endl;

			// find matching piece to the right and shift to the right
			if(columnCursor->edges[columnCursor->rightIndex].isEdgeVar) {
				columnCursor = nullptr;
				cout << "Right edge piece found" << endl;
			} else {
				// find a match
				cout << "Looking for right match" << endl;
				pair<PuzzlePiece*, int> matchPair = columnCursor->match(columnCursor->rightIndex, pieces, numPieces);
				PuzzlePiece *matchingPiece = matchPair.first;
				piecesMatched++;
				matchingPiece->isConnected = true;
				columnCursor->rightNeighbor = matchingPiece;
				matchingPiece->leftNeighbor = columnCursor; // not reallly necessary
				matchingPiece->rightIndex = PuzzlePiece::oppIndex(matchPair.second);

				columnCursor = matchingPiece;
			}
		}

		cout << "Row cursor: piece " << rowCursor->number << endl;

		if(rowCursor->edges[rowCursor->downIndex()].isEdgeVar) {
			rowCursor = nullptr;
			cout << "Bottom edge piece found" << endl;
		} else {
			// find a match
			cout << "Looking for down match" << endl;
			pair<PuzzlePiece*, int> matchPair = rowCursor->match(rowCursor->downIndex(), pieces, numPieces);
			PuzzlePiece *matchingPiece = matchPair.first;
			piecesMatched++;
			matchingPiece->isConnected = true;
			rowCursor->downNeighbor = matchingPiece;
			matchingPiece->upNeighbor = rowCursor; // not reallly necessary
			matchingPiece->rightIndex = PuzzlePiece::nextIndex(matchPair.second);

			rowCursor = matchingPiece;
		}
	}

	cout << "Puzzle completed!" << endl;

	// todo: better evaluation of whether the puzzle is actually completed

	//print:
	// (doesn't work properly bc the main loop goes top to bottom first)
	rowCursor = root;
	while(rowCursor != nullptr) {
		PuzzlePiece *columnCursor = rowCursor;
		while(columnCursor != nullptr) {
			columnCursor->print();
			columnCursor = columnCursor->rightNeighbor;
		}
		cout << endl;
		rowCursor = rowCursor->downNeighbor;
	}

	// display completed puzzle
	namedWindow("temp");

	return 0;
}

// close to 0 is a good match
double EdgeOfPiece::match(EdgeOfPiece other) {

	// prep by rotating one edge (by flipping twice)
	vector<Point> flippedEdge = vector<Point>(other.edge.size());
	for(int i = 0; i < other.edge.size(); i++) {
		flippedEdge[i].x = -other.edge[i].x;
		flippedEdge[i].y = -other.edge[i].y;
	}

	// I don't understand why this needs to be a pointer. otherwise it thinks computeDistance() is a virtual fn.
	// also don't know what's up w the create() function
	Ptr<HausdorffDistanceExtractor> h = createHausdorffDistanceExtractor();
	return h->computeDistance(edge, flippedEdge);

	// return matchShapes(edge, other.edge, CONTOURS_MATCH_I3, 0);
}

bool EdgeOfPiece::isEdge() {
	vector<double> fittedLine; // not clear if this is the right type
	fitLine(edge, fittedLine, DIST_L2, 0.0, .01, .01);

	// can I do this in one line
	double vx = fittedLine[0];
	double vy = fittedLine[1];
	double x0 = fittedLine[2];
	double y0 = fittedLine[3];
	double a = -vy;
	double b = vx;
	double c = vy*x0 - vx*y0;

	double totalDist = 0;
	double denom = sqrt(pow(a, 2) + pow(b, 2));
	for(Point p: edge) {
		// does += or / have precedence? can I do this in one line?
		double singleDist = abs(a*p.x + b*p.y + c) / denom;
		totalDist += singleDist;
	}
	double avgDist = totalDist / edge.size();

	return avgDist < 10;
}

PuzzlePiece::PuzzlePiece() {
	// empty
}

PuzzlePiece::PuzzlePiece(Mat m, int i, bool verbose) {
	img = m;
	number = i+1;
	process(verbose);
}

// todo: break this into steps. first get the piece border, then split into chunks
void PuzzlePiece::process(bool verbose) {
	// check that the image is valid

	Mat grey;
	Mat img_copy = img.clone();
	cvtColor(img_copy, grey, COLOR_BGR2GRAY); // imread stores as BGR

	if(verbose) {
		namedWindow("grey");
		imshow("grey", grey);
		waitKey(0);
	}

	// todo: smooth before thresholding

	// threshold computed for now by looking at histogram in python
	// could compute the threshold by looking at histogram dropoff
	threshold(grey, grey, 30, 255, THRESH_BINARY);

	if(verbose) {
		imshow("grey", grey);
		waitKey(0);
	}

	// cout << "var type: " << grey.type() << endl;

	vector<vector<Point>> contours;
	// todo: try chain_approx_simple
	// CHAIN_APPROX_NONE
	findContours(grey, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// todo: verify that # contours is > 0

	// choose the biggest contour
	outline = contours[0];
	int maxSize = contours[0].size();
	for(int i = 1; i < contours.size(); i++) {
		if(contours[i].size() > maxSize) {
			maxSize = contours[i].size();
			outline = contours[i];
		}
	}

	// put contour into a vector bc drawContours() requires that
	contours.clear();
	contours.push_back(outline);

	Scalar blue(255, 0, 0);
	Scalar red(0, 0, 255);
	Scalar green(0, 255, 0);
	Scalar purple(128, 0, 128);
	if(verbose) {
		// display the outline
		drawContours(img_copy, contours, -1, blue, 5);
		imshow("grey", img_copy);
		waitKey(0);
	}

	// bounding box
	Rect boundingBox = boundingRect(outline);
	if(verbose) {
		rectangle(img_copy, boundingBox.tl(), boundingBox.br(), blue, 5);
		imshow("grey", img_copy);
		waitKey(0);
	}

	//decrease the bounding box to get the "core" rectangle of the piece
	//todo: should have better way to choose increment (chose 50 pixels by trial and error)
	core = boundingBox; // is this a copy? probably not...
//	cout << "bounding box size: " << core.width << " " << core.height << endl;
//	cout << "top left" << core.tl() << endl;
//	cout << "bottom right" << core.br() << endl;

	// for now, I'm not going to worry about the (common) case where all 4 corners are empty.
	// it happens any time there are two opposite nubs.

	// for some reason this was increasing the top left AND bottom right...
	// core = Rect(core.tl() + Point(0, 10), core.br());

	// I guess you can add Points. why?

	// first top
	while(pointPolygonTest(outline, core.tl() + Point(50, 50), false) == -1
			&& pointPolygonTest(outline, core.tl() + Point(core.width - 50, 50), false) == -1) {
		core = Rect(core.tl().x, core.tl().y + 50, core.width, core.height - 50);

		// maybe return false in this case?
		if (core.tl().y > core.br().y) {
			cout << "Failed to identify piece core." << endl;
			return;
		}
	}
	// right
	while(pointPolygonTest(outline, core.tl() + Point(core.width - 50, 50), false) == -1
			&& pointPolygonTest(outline, core.br() + Point(-50, -50), false) == -1) {
		core = Rect(core.tl().x, core.tl().y, core.width - 50, core.height);

		// maybe return false in this case?
		if (core.tl().x > core.br().x) {
			cout << "Failed to identify piece core." << endl;
			return;
		}
	}
	// for these pieces in particular, don't need to check bottom and left.

	if(verbose) {
		// show core
		rectangle(img_copy, core.tl(), core.br(), red, 5);
		imshow("grey", img_copy);
		waitKey(0);
	}

	cout << "num points: " << outline.size() << endl;

	// find closest points to each of the corners of the core
	// (future: identify corners on the contour and find the closest corner)
	int tl_index = 0, tr_index = 0, bl_index = 0, br_index = 0;
	double tl_mindist = 1000, tr_mindist = 1000, bl_mindist = 1000, br_mindist = 1000; // must be better default
	for(int i = 0; i < outline.size(); i++) {

		double tl_dist = norm(core.tl() - outline[i]);
		double tr_dist = norm(core.tl() + Point(core.width, 0) - outline[i]);
		double bl_dist = norm(core.br() + Point(-(core.width), 0) - outline[i]);
		double br_dist = norm(core.br() - outline[i]);

		if (tl_dist < tl_mindist) {
			tl_mindist = tl_dist;
			tl_index = i;
		}
		if (tr_dist < tr_mindist) {
			tr_mindist = tr_dist;
			tr_index = i;
		}
		if (bl_dist < bl_mindist) {
			bl_mindist = bl_dist;
			bl_index = i;
		}
		if (br_dist < br_mindist) {
			br_mindist = br_dist;
			br_index = i;
		}
	}

	// make this a function : constructEdge(start_index, end_index, vector)
	// end index needs the +1?
	// better to pass outline by reference? I don't think I'm passing the data anyway.
	// what if the ordering of the pieces is counter clockwise
	edges[0].edge = constructEdge(outline, tr_index, tl_index);
	edges[1].edge = constructEdge(outline, br_index, tr_index);
	edges[2].edge = constructEdge(outline, bl_index, br_index);
	edges[3].edge = constructEdge(outline, tl_index, bl_index);

	if(verbose) {
		// reset the image and plot the edges
		vector<vector<Point>> edge_vector = {edges[0].edge, edges[1].edge, edges[2].edge, edges[3].edge}; // temp so can plot
		img_copy = img.clone(); // it's pointing to the same data I guess
		drawContours(img_copy, edge_vector, 0, blue, 5);
		drawContours(img_copy, edge_vector, 1, red, 5);
		drawContours(img_copy, edge_vector, 2, green, 5);
		drawContours(img_copy, edge_vector, 3, purple, 5);
		circle(img_copy, outline[tl_index], 5, blue, 15);
		circle(img_copy, outline[tr_index], 5, red, 15);
		circle(img_copy, outline[bl_index], 5, green, 15);
		circle(img_copy, outline[br_index], 5, purple, 15);

		imshow("grey", img_copy);
		waitKey(0);
	}

	for(int i = 0; i < 4; i++) {
		// check if these are actual edges and set isEdgeVar
		edges[i].isEdgeVar = edges[i].isEdge();

		// rotate and translate the edges for easier comparison
		// translate all the edges to line up with the origin
		// (easier to line up edges from images w different dimensions. can translate again to display.)
		if(!edges[i].isEdgeVar) {
			Rect edgeBound = boundingRect(edges[i].edge);
			double mid_x = (edgeBound.tl().x + edgeBound.br().x) / 2;
			double mid_y = (edgeBound.tl().y + edgeBound.br().y) / 2;
			for(Point &p: edges[i].edge) {
				p.x -= mid_x;
				p.y -= mid_y;
			}
		}
	}

	// right edge: rotate by flipping across y = -x, then across y = 0 (origin is upper right)
	if(!edges[1].isEdgeVar) {
		for(Point &p: edges[1].edge) {
			double temp = p.y;
			p.y = -p.x;
			p.x = temp;
		}
	}
	// lower edge: rotate by flipping in x and y direction
	if(!edges[2].isEdgeVar) {
		for(Point &p: edges[2].edge) {
			p.y = -p.y;
			p.x = -p.x;
		}
	}
	// left edge: rotate by flipping across y = x then across y = 0 (origin is upper right)
	if(!edges[3].isEdgeVar) {
		for(Point &p: edges[3].edge) {
			double temp = p.y;
			p.y = p.x;
			p.x = -temp;
		}
	}

	if(verbose) {
			// reset the image and plot the edges
			vector<vector<Point>> edge_vector = {edges[0].edge, edges[1].edge, edges[2].edge, edges[3].edge}; // temp so can plot
			for(int i = 0; i < 4; i++) {
				for(Point &p: edge_vector[i]) {
					p.x += 1000;
					p.y += 1000;
				}
			}
			vector<Scalar> colors = {blue, red, green, purple};
			img_copy = img.clone(); // it's pointing to the same data I guess
			for(int i = 0; i < 4; i++) {
				if(!edges[i].isEdgeVar) drawContours(img_copy, edge_vector, i, colors[i], 5);
			}
			imshow("grey", img_copy);
			waitKey(0);
		}

	if(verbose) {
		destroyWindow("grey");
	}

	cout << "number of points: " << outline.size() << endl;
}

//returns true if 2 or more edges
bool PuzzlePiece::isCorner() {
	return countEdges() >= 2;
}

// implement. also probably call this as part of pocessing and store as numEdges
int PuzzlePiece::countEdges() {
	int count = 0;
	for(int i = 0; i < 4; i++) {
		if(edges[i].isEdgeVar) {
			count++;
		}
	}
	return count;
}

// return index of right connection
// assumption: the piece has exactly 2 consecutive edges
int PuzzlePiece::orientRoot() {
	if(edges[0].isEdgeVar) {
		if(edges[1].isEdgeVar) {
			return 2;
		}
		else return 1;
	}
	else if(edges[1].isEdgeVar) return 3;
	else return 0;
}

// could have done this with mod
int PuzzlePiece::oppIndex(int index) {
	return (index + 2) % 4;
}

int PuzzlePiece::nextIndex(int index) {
	return (index + 1) % 4;
}

int PuzzlePiece::prevIndex(int index) {
	return (index - 1) % 4;
}

int PuzzlePiece::downIndex() {
	return nextIndex(rightIndex);
}

// search through edges to find a match
pair<PuzzlePiece*, int> PuzzlePiece::match(int edgeIndex, PuzzlePiece pieces[], int numPieces) {

	bool firstScore = true;
	double bestMatchScore; // find a better way to set it
	int bestMatchPieceIdx;
	int bestMatchEdgeIdx;

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdgeVar) continue; // skip if it's an edge
			double score = edges[edgeIndex].match(pieces[i].edges[j]);
			cout << "Piece " << number << " scores " << score << " against index " << j << " of piece " << i+1 << endl;
			if(firstScore) {
				bestMatchScore = score;
				bestMatchPieceIdx = i;
				bestMatchEdgeIdx = j;
				firstScore = false;
			}
			else if(score < bestMatchScore) { // low score is best
				bestMatchScore = score;
				bestMatchPieceIdx = i;
				bestMatchEdgeIdx = j;
			}
		}
	}

	cout << "Piece " << number << " matches edge " << bestMatchEdgeIdx << " of piece " << bestMatchPieceIdx+1 << endl;

	return make_pair(&pieces[bestMatchPieceIdx], bestMatchEdgeIdx);
}

void PuzzlePiece::print() {
	cout << setw(4) << number;
}

// note: outline should be oriented counter clockwise bc outer contour. but, I don't think that guarantees sequential order.
vector<Point> PuzzlePiece::constructEdge(vector<Point> outline, int firstIdx, int secondIdx) {
	if(secondIdx > firstIdx) {
		vector<Point> temp(outline.begin() + firstIdx, outline.begin() + secondIdx + 1);
		return temp;
	} else {
		vector<Point> temp(outline.begin() + firstIdx, outline.end());
		temp.insert(temp.end(), outline.begin(), outline.begin() + secondIdx + 1);
		return temp;
	}
}

void displayPuzzle(PuzzlePiece *root) {
	// should first traverse the puzzle to figure out the size needed.

	// todo: create a fake puzzle to test this on so I don't have to run the full code

	double width = root->core.width;
	double height = root->core.height;
	Mat completedPuzzle = Mat::zeros(root->img.size(), root->img.type());

	int row = 0;
	int col = 0;
	PuzzlePiece *rowCursor = root;
	while(rowCursor != nullptr) {
		PuzzlePiece *columnCursor = rowCursor;
		while(columnCursor != nullptr) {

			// output
			// todo: make a copy of img and translate it by col * width or something
			// tried to make color mask but turned out looking blue.
			Mat grey;
			cvtColor(columnCursor->img, grey, COLOR_BGR2GRAY);
			Mat mask = Mat::zeros(grey.size(), grey.type());
			vector<vector<Point>> outlines = {columnCursor->outline};
			drawContours(mask, outlines, -1, 255, -1); // thickness=-1 fills in the contour
			columnCursor->img.copyTo(completedPuzzle, mask);

			columnCursor = columnCursor->rightNeighbor;
			col++;
		}
		cout << endl;
		rowCursor = rowCursor->downNeighbor;
		row++;
	}

	imshow("temp", completedPuzzle);
	waitKey(0);
	destroyWindow("temp");
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
