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
	/*
	string numPiecesStr;
	cout << "How many pieces?" << endl;
	cin >> numPiecesStr;
	int numPieces = stoi(numPiecesStr);
	*/

	bool process_verbose = false;
	bool match_verbose = false;

	int numPieces = 16;
	cout << numPieces << " pieces" << endl;

	// load images
	// todo: check if the file loaded properly
	Mat images[numPieces];
	string dir = "/Users/blakechellew/Documents/Code/workspace/PuzzleSolver2/Pieces_16_green/Piece";
	for( int i = 0; i < numPieces; i++) {
		string filename = dir + to_string(i+1) + ".jpeg";
		images[i] = imread(filename);
	}

	// try displaying a piece
	namedWindow("temp");
	imshow("temp", images[0]);
	waitKey(0);
	destroyWindow("temp");

	// test: process one piece
	// PuzzlePiece p = PuzzlePiece(images[8], 8, process_verbose);
	// exit(0);

	// todo: if the piece construction fails, stop the program
	PuzzlePiece pieces[numPieces];
	for(int i = 0; i < numPieces; i++) {
		pieces[i] = PuzzlePiece(images[i], i, process_verbose); // last argument is "verbose"
	}

	cout << "checkpoint" << endl;

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

	// test a specific edge
	int testIndex = 6;
	pieces[testIndex-1].isConnected=true;
	pieces[testIndex-1].rightIndex = 0;
	// pieces[testIndex-1].match(pieces[testIndex-1].rightIndex, pieces, numPieces, true);
	// close matches:
	pieces[testIndex-1].edges[0].match(pieces[13].edges[3], true); // score 64
	pieces[testIndex-1].edges[0].match(pieces[15].edges[0], true); // correct match, score 214!
	// and look at the closest matches:
//	pieces[testIndex-1].edges[1].match(pieces[5].edges[2], true); // correct match, score 59
//	pieces[testIndex-1].edges[1].match(pieces[2].edges[0], true); // score 65
//	pieces[testIndex-1].edges[1].match(pieces[6].edges[1], true); // score 49
//	pieces[testIndex-1].edges[1].match(pieces[11].edges[0], true);  // score 46
//	pieces[testIndex-1].edges[1].match(pieces[11].edges[2], true);  // score 48
	exit(0);

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

	/*
	// test: create a fake puzzle for display
	PuzzlePiece *root2 = &pieces[firstCornerIdx];
	root2->rightIndex = 1;
	root2->rightNeighbor = &pieces[4];
	root2->downNeighbor = &pieces[0];
	pieces[4].rightIndex = 0;
	pieces[4].rightNeighbor = &pieces[6];
	pieces[4].leftNeighbor = &pieces[firstCornerIdx];
	pieces[6].rightIndex = 0;
	pieces[6].rightNeighbor = &pieces[7];
	pieces[6].leftNeighbor = &pieces[4];
	pieces[7].rightIndex = 0;
	pieces[7].leftNeighbor = &pieces[6];
	pieces[0].rightIndex = 3;
	pieces[0].rightNeighbor = &pieces[13];
	pieces[0].downNeighbor = &pieces[1];
	pieces[0].upNeighbor = &pieces[firstCornerIdx];
	pieces[13].rightIndex = 1;
	pieces[13].rightNeighbor = &pieces[5];
	pieces[13].leftNeighbor = &pieces[0];
	pieces[5].rightIndex = 0;
	pieces[5].rightNeighbor = &pieces[14];
	pieces[5].leftNeighbor = &pieces[13];
	pieces[14].rightIndex = 1;
	pieces[14].leftNeighbor = &pieces[5];
	pieces[1].rightIndex = 3;
	pieces[1].upNeighbor = &pieces[0];
	pieces[1].rightNeighbor = &pieces[11];
	pieces[1].downNeighbor = &pieces[8];
	pieces[11].rightIndex = 2;
	pieces[11].leftNeighbor = &pieces[1];
	pieces[11].rightNeighbor = &pieces[15];
	pieces[15].rightIndex = 1;
	pieces[15].leftNeighbor = &pieces[11];
	pieces[15].rightNeighbor = &pieces[12];
	pieces[12].rightIndex = 3;
	pieces[12].leftNeighbor = &pieces[15];
	pieces[8].rightIndex = 1;
	pieces[8].upNeighbor = &pieces[1];
	pieces[8].rightNeighbor = &pieces[9];
	pieces[9].rightIndex = 3;
	pieces[9].leftNeighbor = &pieces[8];
	pieces[9].rightNeighbor = &pieces[2];
	pieces[2].rightIndex = 2;
	pieces[2].leftNeighbor = &pieces[9];
	pieces[2].rightNeighbor = &pieces[10];
	pieces[10].rightIndex = 0;
	pieces[10].leftNeighbor = &pieces[2];
	displayPuzzle(root2, true);
	exit(0);
	*/

	PuzzlePiece *root = &pieces[firstCornerIdx];
	root->isConnected = true;
	// todo: check if pieces matched reaches numPieces
	int piecesMatched = 1;
	// set the orientation of the root piece
	root->rightIndex = root->orientRoot();

	cout << "Start matching" << endl;

	// wonder if there is a cleaner way to write these loops e.g. w less duplication

	PuzzlePiece *rowCursor = root;

	while(rowCursor != nullptr) {

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
				pair<PuzzlePiece*, int> matchPair = columnCursor->match(columnCursor->rightIndex, pieces, numPieces, match_verbose);
				if(matchPair.first == nullptr) {
					columnCursor = nullptr;
					break;
				}
				PuzzlePiece *matchingPiece = matchPair.first;
				piecesMatched++;
				cout << "pieces matched: " << piecesMatched << endl;
				matchingPiece->isConnected = true;
				columnCursor->rightNeighbor = matchingPiece;
				matchingPiece->leftNeighbor = columnCursor; // not really necessary
				matchingPiece->rightIndex = PuzzlePiece::oppIndex(matchPair.second);

				columnCursor = matchingPiece;
			}
		}

		if(piecesMatched >= numPieces) break;  // loop above could increment piecesMatched

		cout << "Row cursor: piece " << rowCursor->number << endl;

		if(rowCursor->edges[rowCursor->downIndex()].isEdgeVar) {
			rowCursor = nullptr;
			cout << "Bottom edge piece found" << endl;
		} else {
			// find a match
			cout << "Looking for down match" << endl;
			pair<PuzzlePiece*, int> matchPair = rowCursor->match(rowCursor->downIndex(), pieces, numPieces, match_verbose);
			if(matchPair.first == nullptr) {
				rowCursor = nullptr;
				break;
			}
			PuzzlePiece *matchingPiece = matchPair.first;
			piecesMatched++;
			cout << "pieces matched: " << piecesMatched << endl;
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

	// display completed puzzle:
	displayPuzzle(root, true);

	return 0;
}

double PuzzlePiece::scalingLength = 0;

// close to 0 is a good match
double EdgeOfPiece::match(EdgeOfPiece other, bool verbose) {

	// prep by rotating one edge (by flipping twice)
	vector<Point> flippedEdge = vector<Point>(other.edge.size());
	for(int i = 0; i < other.edge.size(); i++) {
		flippedEdge[i].x = -other.edge[i].x;
		flippedEdge[i].y = -other.edge[i].y;
	}

	// show the edges being compared
	Mat blank = Mat::zeros(3000, 3000, CV_8UC3);
	vector<vector<Point>> twoEdges = {edge, flippedEdge};
	for(Point &p: twoEdges[0]) {
		p.x += 1000;
		p.y += 1000;
	}
	for(Point &p: twoEdges[1]) {
		p.x += 1000;
		p.y += 1000;
	}

	if(verbose) {
		namedWindow("compare edges");
		// drawContours(blank, twoEdges, 0, Scalar(255, 0, 0), 5);
		// drawContours(blank, twoEdges, 1, Scalar(0, 0, 255), 5);
		for(Point p: twoEdges[0]) {
			circle(blank, p, 5, Scalar(255, 0, 0), 10);
		}
		for(Point p: twoEdges[1]) {
			circle(blank, p, 5, Scalar(0, 0, 255), 10);
		}
		imshow("compare edges", blank);
		waitKey(0);
		destroyWindow("compare edges");
	}

	// I don't understand why this needs to be a pointer. otherwise it thinks computeDistance() is a virtual fn.
	// also don't know what's up w the create() function
	// Ptr<HausdorffDistanceExtractor> h = createHausdorffDistanceExtractor();
	// return h->computeDistance(edge, flippedEdge);

	return max(edgeComparisonScore(edge, flippedEdge), edgeComparisonScore(flippedEdge, edge));

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

	Scalar blue(255, 0, 0);
	Scalar red(0, 0, 255);
	Scalar green(0, 255, 0);
	Scalar purple(128, 0, 128);
	vector<Scalar> colors = {blue, red, green, purple};

	// identify background color
	Mat img_hsv;  // possible to display?
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	vector<Vec3b> backgroundColors;
	int cornerSize = 50;
	Mat topLeftCorner = img_hsv(Rect(0, 0, cornerSize, cornerSize));
	Mat topRightCorner = img_hsv(Rect(img.size().width-cornerSize, 0, cornerSize, cornerSize));
	Mat bottomLeftCorner = img_hsv(Rect(0, img.size().height-cornerSize, cornerSize, cornerSize));
	Mat bottomRightCorner = img_hsv(Rect(img.size().width-cornerSize, img.size().height-cornerSize, cornerSize, cornerSize));
	for(MatIterator_<Vec3b> it = topLeftCorner.begin<Vec3b>(), end=topLeftCorner.end<Vec3b>(); it != end; it++) {
		backgroundColors.push_back(*it);
	}
	for(MatIterator_<Vec3b> it = topRightCorner.begin<Vec3b>(), end=topRightCorner.end<Vec3b>(); it != end; it++) {
		backgroundColors.push_back(*it);
	}
	for(MatIterator_<Vec3b> it = bottomLeftCorner.begin<Vec3b>(), end=bottomLeftCorner.end<Vec3b>(); it != end; it++) {
		backgroundColors.push_back(*it);
	}
	for(MatIterator_<Vec3b> it = bottomRightCorner.begin<Vec3b>(), end=bottomRightCorner.end<Vec3b>(); it != end; it++) {
		backgroundColors.push_back(*it);
	}
	double h_channel_min = backgroundColors[0][0];
	double h_channel_max = backgroundColors[0][0];
	double s_channel_min = backgroundColors[0][1];
	double s_channel_max = backgroundColors[0][1];
	double v_channel_min = backgroundColors[0][2];
	double v_channel_max = backgroundColors[0][2];
	for(Scalar c: backgroundColors) {
		if(c[0] < h_channel_min) h_channel_min = c[0];
		if(c[0] > h_channel_max) h_channel_max = c[0];
		if(c[1] < s_channel_min) s_channel_min = c[1];
		if(c[1] > s_channel_max) s_channel_max = c[1];
		if(c[2] < v_channel_min) v_channel_min = c[2];
		if(c[2] > v_channel_max) v_channel_max = c[2];
	}
	double h_channel_width = h_channel_max - h_channel_min;
	double s_channel_width = s_channel_max - s_channel_min;
	double v_channel_width = v_channel_max - v_channel_min;

	double hueBuffer = 0.5;  // fraction of color range that is added to each end
	double satBuffer = 1.5;
	double valueBuffer = 1.5;
	// Scalar colorLowerBound = Scalar(max(0.0, h_channel_min - h_channel_width/colorRangeBuffer), max(0.0, s_channel_min - s_channel_width/colorRangeBuffer), max(0.0, v_channel_min - v_channel_width/colorRangeBuffer));
	// Scalar colorUpperBound = Scalar(min(255.0, h_channel_max + h_channel_width/colorRangeBuffer), min(255.0, s_channel_max + s_channel_width/colorRangeBuffer), min(255.0, v_channel_max + v_channel_width/colorRangeBuffer));
	Scalar colorLowerBound = Scalar(max(0.0, h_channel_min - h_channel_width/hueBuffer), max(0.0, s_channel_min - s_channel_width/satBuffer), max(0.0, v_channel_min - v_channel_width/valueBuffer));
	Scalar colorUpperBound = Scalar(min(255.0, h_channel_max + h_channel_width/hueBuffer), min(255.0, s_channel_max + s_channel_width/satBuffer), min(255.0, v_channel_max + v_channel_width/valueBuffer));
	cout << "value range: " << colorLowerBound << " " << colorUpperBound << endl;

	// Mat blurredImage;
	// blur(img, blurredImage, Size(20, 20));
	// cout << "color bounds: " << colorLowerBound << " " << colorUpperBound << endl;
	Mat color_mask;
	inRange(img_hsv, colorLowerBound, colorUpperBound, color_mask);
	color_mask = 255 - color_mask;  // invert
	if(verbose) {
		namedWindow("grey");
		imshow("grey", color_mask);
		waitKey(0);
	}
	Mat close_kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
	filter2D(color_mask, color_mask, -1, close_kernel);
	if(verbose) {
		imshow("grey", color_mask);
		waitKey(0);
	}

	vector<vector<Point>> contours;
	findContours(color_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// alt: CHAIN_APPROX_SIMPLE

	if(contours.size() < 2) {  // piece and coin
		cout << "ERROR: no puzzle piece found" << endl;
		return;
	}

	Mat img_copy = img.clone();
	if(verbose) {
		cout << "display all contours" << endl;
		drawContours(img_copy, contours, -1, blue, 5);
		imshow("grey", img_copy);
		waitKey(0);
	}

	// choose the biggest contours
	outline = contours[0];
	vector<Point> coin = contours[1];
	double maxSize = contourArea(contours[0]);
	double secondMaxSize = contourArea(contours[1]);
	for(int i = 1; i < contours.size(); i++) {
		double currentArea = contourArea(contours[i]);
		if(currentArea > maxSize) {
			secondMaxSize = maxSize;
			coin = outline;
			maxSize = currentArea;
			outline = contours[i];
		}
		else if(currentArea > secondMaxSize) {
			secondMaxSize = currentArea;
			coin = contours[i];
		}
	}

	// show all the contours and print their areas
	/*
	img_copy = img.clone();
	for(int i = 0; i < contours.size(); i++) {
		cout << contourArea(contours[i]) << endl;
		drawContours(img_copy, contours, i, blue, 5);
		imshow("grey", img_copy);
		waitKey(0);
	}
	*/

	// check which one is the circle
	double outlineArea = contourArea(outline);
	double coinArea = contourArea(coin);
	Point2f coinCenter;
	Point2f outlineCenter;
	float outlineRadius;  // making this type double didn't work
	float coinRadius;
	double pi = atan(1)*4;
	minEnclosingCircle(outline, outlineCenter, outlineRadius);
	minEnclosingCircle(coin, coinCenter, coinRadius);
	double coinCircleArea = pi * pow(coinRadius, 2);
	double outlineCircleArea = pi * pow(outlineRadius, 2);
	// swap outline and circle if needed
	if(coinCircleArea / coinArea > outlineCircleArea / outlineArea) {
		vector<Point> temp = coin;
		coin = outline;
		outline = temp;
	}

	// scale everything up to regular size
	// also include the scale factor
	if(scalingLength == 0) {
		scalingLength = coinRadius;
	}
	double normalize_factor = scalingLength / coinRadius;
	cout << "Piece " << number << " scaling by " << normalize_factor << endl;
	for(Point &p: outline) {
		p.x *= normalize_factor;
		p.y *= normalize_factor;
	}
	for(Point &p: coin) {  // just for debug purposes
		p.x *= normalize_factor;
		p.y *= normalize_factor;
	}
	// resize the original image
	// not sure if this will reallocate properly
	resize(img, img, Size(img.size[1] * normalize_factor, img.size[0] * normalize_factor));

	/*
	// smooth out the contour
	approxPolyDP(outline, outline, 30, true);

	// output outline to file
	ofstream outlineFile("piece_outline.csv");
	for(Point p: outline) {
		outlineFile << p.x << "," << p.y << "\n";
	}
	outlineFile.close();
	exit(0);
	*/

	// put contour into a vector bc drawContours() requires that
	contours.clear();
	contours.push_back(outline);
	contours.push_back(coin);

	if(verbose) {
		// display the outline
		img_copy = img.clone();
		drawContours(img_copy, contours, 0, blue, 5);
		drawContours(img_copy, contours, 1, red, 5);
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

	// set scan params based on size of bounding box.
	// solves issue where images at different resolution were handled differently

	// cout << "width " << boundingBox.width << ", height " << boundingBox.height << endl;
	int scanWidth = max(boundingBox.width, boundingBox.height) / 16;
	int scanDepth = max(boundingBox.width, boundingBox.height) / 40;

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

	// todo: stopping conditions if things go wrong e.g.
	// todo: too much copy / paste of code here

	// scan from top
	double top_buffer = 0;
	vector<Point> scanLine;
	for(int i = 0; i < core.width / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(i * scanWidth, 0));
	}
	bool scanEnd = false;
	do {
		double leftIndex = 0;
		double rightIndex = scanLine.size() - 1;
		for(; leftIndex < scanLine.size(); leftIndex++) {
			if(pointPolygonTest(outline, scanLine[leftIndex], false) == 1) {
				break;
			}
		}
		for(; rightIndex >= 0; rightIndex--) {
			if(pointPolygonTest(outline, scanLine[rightIndex], false) == 1) {
				break;
			}
		}

		// check if ratio is > 50%
		double occupiedRatio = (rightIndex - leftIndex) / scanLine.size();
		// cout << leftIndex << " " << rightIndex << " " << occupiedRatio << endl;
		if(occupiedRatio > 0.5) {
			scanEnd = true;
		} else {
			for(Point &p: scanLine) {
				p.y += scanDepth;
			}
			top_buffer += scanDepth;
		}

		if(top_buffer > core.height) {
			cout << "ERROR: unable to identify piece core" << endl;
			break;
		}
	} while(!scanEnd);

	core = Rect(core.tl().x, core.tl().y + top_buffer, core.width, core.height - top_buffer);

	// scan from bottom
	double bottom_buffer = 0;
	scanLine.clear();
	for(int i = 0; i < core.width / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(i * scanWidth, core.height));
	}
	scanEnd = false;
	do {
		double leftIndex = 0;
		double rightIndex = scanLine.size() - 1;
		for(; leftIndex < scanLine.size(); leftIndex++) {
			if(pointPolygonTest(outline, scanLine[leftIndex], false) == 1) {
				break;
			}
		}
		for(; rightIndex >= 0; rightIndex--) {
			if(pointPolygonTest(outline, scanLine[rightIndex], false) == 1) {
				break;
			}
		}

		// check if ratio is > 50%
		double occupiedRatio = (rightIndex - leftIndex) / scanLine.size();
		// cout << leftIndex << " " << rightIndex << " " << occupiedRatio << endl;
		if(occupiedRatio > 0.5) {
			scanEnd = true;
		} else {
			for(Point &p: scanLine) {
				p.y -= scanDepth;
			}
			bottom_buffer += scanDepth;
		}

		if(bottom_buffer > core.height) {
			cout << "ERROR: unable to identify piece core" << endl;
			break;
		}
	} while(!scanEnd);

	core = Rect(core.tl().x, core.tl().y, core.width, core.height - bottom_buffer);

	// scan from left
	double left_buffer = 0;
	scanLine.clear();
	for(int i = 0; i < core.height / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(0, i * scanWidth));
	}
	scanEnd = false;
	do {
		double leftIndex = 0;
		double rightIndex = scanLine.size() - 1;
		for(; leftIndex < scanLine.size(); leftIndex++) {
			if(pointPolygonTest(outline, scanLine[leftIndex], false) == 1) {
				break;
			}
		}
		for(; rightIndex >= 0; rightIndex--) {
			if(pointPolygonTest(outline, scanLine[rightIndex], false) == 1) {
				break;
			}
		}

		// check if ratio is > 50%
		double occupiedRatio = (rightIndex - leftIndex) / scanLine.size();
		cout << leftIndex << " " << rightIndex << " " << occupiedRatio << endl;
		if(occupiedRatio > 0.5) {
			scanEnd = true;
		} else {
			for(Point &p: scanLine) {
				p.x += scanDepth;
			}
			left_buffer += scanDepth;
		}

		if(left_buffer > core.width) {
			cout << "ERROR: unable to identify piece core" << endl;
			break;
		}
	} while(!scanEnd);

	core = Rect(core.tl().x + left_buffer, core.tl().y, core.width - left_buffer, core.height);

	// scan from right
	double right_buffer = 0;
	scanLine.clear();
	for(int i = 0; i < core.height / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(core.width, i * scanWidth));
	}
	scanEnd = false;
	do {
		double leftIndex = 0;
		double rightIndex = scanLine.size() - 1;
		for(; leftIndex < scanLine.size(); leftIndex++) {
			if(pointPolygonTest(outline, scanLine[leftIndex], false) == 1) {
				break;
			}
		}
		for(; rightIndex >= 0; rightIndex--) {
			if(pointPolygonTest(outline, scanLine[rightIndex], false) == 1) {
				break;
			}
		}

		// check if ratio is > 50%
		double occupiedRatio = (rightIndex - leftIndex) / scanLine.size();
		cout << leftIndex << " " << rightIndex << " " << occupiedRatio << endl;
		if(occupiedRatio > 0.5) {
			scanEnd = true;
		} else {
			for(Point &p: scanLine) {
				p.x -= scanDepth;
			}
			right_buffer += scanDepth;
		}

		if(right_buffer > core.width) {
			cout << "ERROR: unable to identify piece core" << endl;
			break;
		}
	} while(!scanEnd);

	core = Rect(core.tl().x, core.tl().y, core.width - right_buffer, core.height);

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

//	for(int i = 0; i < 4; i++) {  // make this part of constructEdge?
//		Moments m = moments(edges[i].edge);
//		int cx = int(m.m10/m.m00);
//		int cy = int(m.m01/m.m00);
//		edges[i].centroid = Point(cx, cy);
//	}
	edges[0].centroid = Point(core.tl().x + core.width/2, core.tl().y);  // Point data type?
	edges[1].centroid = Point(core.br().x, core.tl().y + core.height/2);
	edges[2].centroid = Point(core.tl().x + core.width/2, core.br().y);
	edges[3].centroid = Point(core.tl().x, core.tl().y + core.height/2);

	// todo: verify that these edges are reasonable e.g. have more than a couple points

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
			// shift the edges for easier plotting
			for(int i = 0; i < 4; i++) {
				for(Point &p: edge_vector[i]) {  // apparently contents of edge_vector are copies?
					p.x += 1000;
					p.y += 1000;
				}
			}
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
pair<PuzzlePiece*, int> PuzzlePiece::match(int edgeIndex, PuzzlePiece pieces[], int numPieces, bool verbose) {

	bool firstMatch = true;
	double bestMatchScore; // find a better way to set it
	int bestMatchPieceIdx;
	int bestMatchEdgeIdx;

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdgeVar) continue; // skip if it's an edge
			double score = edges[edgeIndex].match(pieces[i].edges[j], verbose);
			cout << "Piece " << number << " scores " << score << " against index " << j << " of piece " << i+1 << endl;
			if(firstMatch) {
				bestMatchScore = score;
				bestMatchPieceIdx = i;
				bestMatchEdgeIdx = j;
				firstMatch = false;
			}
			else if(score < bestMatchScore) { // low score is best
				bestMatchScore = score;
				bestMatchPieceIdx = i;
				bestMatchEdgeIdx = j;
			}
		}
	}

	if(firstMatch) {
		cout << "ERROR: remaining pieces are edges only." << endl;
		return make_pair(nullptr, -1);
	}

	cout << "Piece " << number << " matches edge " << bestMatchEdgeIdx << " of piece " << bestMatchPieceIdx+1 << endl;
	return make_pair(&pieces[bestMatchPieceIdx], bestMatchEdgeIdx);
}

void PuzzlePiece::print() {
	cout << setw(4) << number;
}

double PuzzlePiece::rotationAngle() {
	if(rightIndex == 1) return 0;
	else if(rightIndex == 2) return 90;
	else if(rightIndex == 3) return 180;
	else return 270;
}

// flips width and height if piece is rotated
double PuzzlePiece::width() {
	if(rotationAngle() == 0 || rotationAngle() == 180) return core.width;
	return core.height;
}

double PuzzlePiece::height() {
	if(rotationAngle() == 0 || rotationAngle() == 180) return core.height;
	return core.width;
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

void displayPuzzle(PuzzlePiece *root, bool verbose, bool checkRotation) {

	namedWindow("temp");

	// traverse the puzzle to figure out the size needed.
	double width = root->width();
	double height = root->height();
	cout << "width: " << width << endl;
	cout << "height: " << height << endl;
	int numPiecesX = 0;
	int numPiecesY = 0;
	PuzzlePiece *cursor = root;
	while(cursor != nullptr) {
		numPiecesY++;
		cursor = cursor->downNeighbor;
	}
	cursor = root;
	while(cursor != nullptr) {
		numPiecesX++;
		cursor = cursor->rightNeighbor;
	}
	cout << "num pieces: " << numPiecesX << " by " << numPiecesY << endl;
	double edgeLength = max(numPiecesX * width, numPiecesY * height); // leave room for rotating at the end
	Mat completedPuzzle = Mat::zeros(edgeLength, edgeLength, root->img.type());
	cout << "completed puzzle size: " << completedPuzzle.size() << endl;

	// loop through the pieces and copy to completed image
	int row = 0;
	int col = 0;
	PuzzlePiece *rowCursor = root;
	while(rowCursor != nullptr) {
		PuzzlePiece *columnCursor = rowCursor;
		col = 0;
		while(columnCursor != nullptr) {

			// todo: the rotation moves the top left corner if pieces are not square. need to account for that.
			// use the new top-left corner in the calculations instead.

			// if attaching left: rotate the left edge centroid. then line it up with the (rotated) centroid of the piece to the left.

			// transformations: rotate and shift the puzzle piece
			Mat img_copy = columnCursor->img.clone();
			Mat transformed = Mat::zeros(completedPuzzle.size(), completedPuzzle.type());
			if(verbose) {
				cout << "show image" << endl;
				imshow("temp", columnCursor->img);
				waitKey(0);
			}
			cout << "top left coordinate: " << columnCursor->core.tl() << endl;
			cout << "x shift: " << col * width - columnCursor->core.tl().x << endl;
			cout << "y shift: " << row * height - columnCursor->core.tl().y << endl;
			// rotate about the center of the piece
			Point rotationCenter = Point(columnCursor->core.tl().x + columnCursor->core.width/2, columnCursor->core.tl().y + columnCursor->core.height/2);
			Mat t1 = getRotationMatrix2D(rotationCenter, columnCursor->rotationAngle(), 1);
			cout << "rotation matrix: " << t1 << endl;
			warpAffine(img_copy, img_copy, t1, img_copy.size());
			if(verbose) {
				cout << "after just rotation:" << endl;
				imshow("temp", img_copy);
				waitKey(0);
			}
			// rotate the centroids of the current piece
			for(int i = 0; i < 4; i++) {
				double cx = columnCursor->edges[i].centroid.x;
				double cy = columnCursor->edges[i].centroid.y;
				columnCursor->edges[i].centroid.x = t1.at<double>(0, 0) * cx + t1.at<double>(0, 1) * cy + t1.at<double>(0, 2);
				columnCursor->edges[i].centroid.y = t1.at<double>(1, 0) * cx + t1.at<double>(1, 1) * cy + t1.at<double>(1, 2);
			}

			// now translate
			// should do in one step; I'm counting on no clipping after the rotation
			// note: could do this check implicitly by waiting until this while loop exits to the other one
			Point2f shift; // about converting int to double if I just do Point

			if(columnCursor->leftNeighbor == nullptr && columnCursor->upNeighbor == nullptr) {  // this is top left corner
				if(columnCursor->rotationAngle() == 90 || columnCursor->rotationAngle() == 270) {
					cout << "shifting TL corner bc of rotation" << endl;
					Point rotated_tl = columnCursor->core.tl()
							+ Point(columnCursor->core.width/2 - columnCursor->core.height/2,
									columnCursor->core.height/2 - columnCursor->core.width/2);
					// cout << "new TL corner: " << rotated_tl << endl;
					shift = -rotated_tl;
				} else {
					shift = -columnCursor->core.tl();
				}
			} else if(columnCursor->leftNeighbor == nullptr) {  // this is left edge
				Point prevCentroid = columnCursor->upNeighbor->edges[columnCursor->upNeighbor->downIndex()].centroid;
				Point currentCentroid = columnCursor->edges[PuzzlePiece::oppIndex(columnCursor->downIndex())].centroid;
				shift = prevCentroid - currentCentroid;
			} else {  // most pieces
				Point prevCentroid = columnCursor->leftNeighbor->edges[columnCursor->leftNeighbor->rightIndex].centroid;
				Point currentCentroid = columnCursor->edges[PuzzlePiece::oppIndex(columnCursor->rightIndex)].centroid;
				shift = prevCentroid - currentCentroid;
			}

			double t_values[] = {1, 0, shift.x,
												0, 1, shift.y};
			Mat t2 = Mat(2, 3, DataType<double>::type, t_values); // not sure about that data type
			cout << "translation matrix: " << t2 << endl;
			warpAffine(img_copy, transformed, t2, transformed.size());
			if(verbose) {
				cout << "show completed transformation" << endl;
				imshow("temp", transformed);
				waitKey(0);
			}

			// also need to shift AND rotate the outline
			vector<Point> shifted_outline = columnCursor->outline;  // verify how this assignment works
			// shifted_outline += Point(col * width - columnCursor->core.tl().x, row * height - columnCursor->core.tl().y);
			for(Point &p: shifted_outline) {
				double temp_x = p.x;
				p.x = t1.at<double>(0, 0) * p.x + t1.at<double>(0, 1) * p.y + t1.at<double>(0, 2) + shift.x;
				p.y = t1.at<double>(1, 0) * temp_x + t1.at<double>(1, 1) * p.y + t1.at<double>(1, 2) + shift.y;
			}
			// and shift the centroids
			for(int i = 0; i < 4; i++) {
				columnCursor->edges[i].centroid.x += shift.x;
				columnCursor->edges[i].centroid.y += shift.y;
			}

			// copy the data within the piece outline to the final image
			Mat mask = Mat::zeros(transformed.size(), transformed.type());
			vector<vector<Point>> outlines = {shifted_outline};
			drawContours(mask, outlines, -1, Scalar(255, 255, 255), -1); // thickness=-1 fills in the contour
			if(verbose) {
				cout << "show mask" << endl;
				imshow("temp", mask);
				waitKey(0);
			}
			transformed.copyTo(completedPuzzle, mask);
			if(verbose) {
				cout << "show completed puzzle with new piece" << endl;
				imshow("temp", completedPuzzle);
				waitKey(0);
			}

			columnCursor = columnCursor->rightNeighbor;
			col++;
		}
		cout << endl;
		rowCursor = rowCursor->downNeighbor;
		row++;
	}

	if(checkRotation) cout << "Enter degrees of clockwise rotation" << endl;

	// show completed puzzle
	imshow("temp", completedPuzzle);
	waitKey(0);

	if(checkRotation) {
		string rotationStr;
		cin >> rotationStr; // not sure how this interacts with waitKey()
		int fullPuzzleRotation = stoi(rotationStr); // error handling
		Point puzzleCenter = Point((numPiecesX * height) / 2, (numPiecesY * width) / 2);
		Mat t3 = getRotationMatrix2D(puzzleCenter, fullPuzzleRotation, 1);
		Mat rotatedPuzzle = Mat::zeros(completedPuzzle.size(), completedPuzzle.type()); // don't want any remnants after the rotation
		warpAffine(completedPuzzle, rotatedPuzzle, t3, rotatedPuzzle.size());
		imshow("temp", rotatedPuzzle);
		waitKey(0);
	}

	destroyWindow("temp");
}

// should reverse the loop also to be more like hausdorff
double edgeComparisonScore(vector<Point> edge1, vector<Point> edge2) {

	// ignore points past the edge
	double minX1 = edge1[0].x;
	double maxX1 = edge1[0].x;
	for(Point p1: edge1) {
		if (p1.x < minX1) minX1 = p1.x;
		if (p1.x > maxX1) maxX1 = p1.x;
	}
	double minX2 = edge1[0].x;
	double maxX2 = edge1[0].x;
	for(Point p2: edge2) {
		if (p2.x < minX2) minX2 = p2.x;
		if (p2.x > maxX2) maxX2 = p2.x;
	}
	double minX = max(minX1, minX2);
	double maxX = min(maxX1, maxX2);

	// loop to find max distance
	double maxDistance = 0;
	for(Point p1: edge1) {
		if(p1.x < minX || p1.x > maxX) continue;
		double minDistance = sqrt(pow(p1.x - edge2[0].x, 2) + pow(p1.y - edge2[0].y, 2));
		for(Point p2: edge2) {
			if(p2.x < minX || p2.x > maxX) continue;
			double distance = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
			if(distance < minDistance) {
				minDistance = distance;
				// cout << "new min distance: " << distance << endl;
			}
		}
		if (minDistance > maxDistance) {
			maxDistance = minDistance;
			// cout << "new biggest min distance: " << maxDistance << endl;
		}
	}
	return maxDistance;
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
