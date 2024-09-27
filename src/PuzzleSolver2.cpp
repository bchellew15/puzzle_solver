#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> //for setw(n)
#include <cmath>
#include <chrono> // for timer
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
	string dir = "/Users/blakechellew/Documents/Code/workspace/PuzzleSolver2/Pieces_16_darkgreen/Piece";
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
	// PuzzlePiece p1 = PuzzlePiece(images[5], 5, true);
	// PuzzlePiece p2 = PuzzlePiece(images[14], 5, true);

	// todo: if the piece construction fails, stop the program
	PuzzlePiece pieces[numPieces];
	for(int i = 0; i < numPieces; i++) {
		pieces[i] = PuzzlePiece(images[i], i); // last argument is "verbose"
	}

	/*
	// test: create a fake puzzle for display
	Puzzle testPuzzle = Puzzle(numPieces, pieces);
	testPuzzle.rows = 4;
	testPuzzle.columns = 4;
	testPuzzle.process();

	testPuzzle.completedPuzzle.push_back(vector<PuzzlePiece*>(4));
	testPuzzle.completedPuzzle.push_back(vector<PuzzlePiece*>(4));
	testPuzzle.completedPuzzle.push_back(vector<PuzzlePiece*>(4));
	testPuzzle.completedPuzzle.push_back(vector<PuzzlePiece*>(4));
	testPuzzle.completedPuzzle[0][0] = &pieces[3];
	testPuzzle.completedPuzzle[0][1] = &pieces[4];
	testPuzzle.completedPuzzle[0][2] = &pieces[6];
	testPuzzle.completedPuzzle[0][3] = &pieces[7];
	testPuzzle.completedPuzzle[1][0] = &pieces[0];
	testPuzzle.completedPuzzle[1][1] = &pieces[13];
	testPuzzle.completedPuzzle[1][2] = &pieces[5];
	testPuzzle.completedPuzzle[1][3] = &pieces[14];
	testPuzzle.completedPuzzle[2][0] = &pieces[1];
	testPuzzle.completedPuzzle[2][1] = &pieces[11];
	testPuzzle.completedPuzzle[2][2] = &pieces[15];
	testPuzzle.completedPuzzle[2][3] = &pieces[12];
	testPuzzle.completedPuzzle[3][0] = &pieces[8];
	testPuzzle.completedPuzzle[3][1] = &pieces[9];
	testPuzzle.completedPuzzle[3][2] = &pieces[2];
	testPuzzle.completedPuzzle[3][3] = &pieces[10];

	pieces[3].rightIndex = 1;
	pieces[4].rightIndex = 0;
	pieces[6].rightIndex = 0;
	pieces[7].rightIndex = 0;
	pieces[0].rightIndex = 3;
	pieces[13].rightIndex = 1;
	pieces[5].rightIndex = 0;
	pieces[14].rightIndex = 1;
	pieces[1].rightIndex = 3;
	pieces[11].rightIndex = 2;
	pieces[15].rightIndex = 1;
	pieces[12].rightIndex = 3;
	pieces[8].rightIndex = 1;
	pieces[9].rightIndex = 3;
	pieces[2].rightIndex = 2;
	pieces[10].rightIndex = 0;

	testPuzzle.display(true);
	exit(0);
	*/

	// create a Puzzle
	Puzzle myPuzzle = Puzzle(numPieces, pieces);
	myPuzzle.process(process_verbose);

	/*
	// time edge comparison
	chrono::time_point<chrono::steady_clock> start_time = chrono::steady_clock::now();
	Puzzle::matchEdges(pieces[0].edges[0], pieces[1].edges[0]);
	chrono::time_point<chrono::steady_clock> end_time = chrono::steady_clock::now();
	// I don't understand these time functions at all
	cout << "Runtime (ms): " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << endl;
	*/

	/*
	// test a specific edge
	cout << Puzzle::matchEdges(pieces[1].edges[3], pieces[11].edges[0], true).score << endl;
	cout << Puzzle::matchEdges(pieces[13].edges[2], pieces[11].edges[1], true).score << endl;

	cout << Puzzle::matchEdges(pieces[1].edges[3], pieces[11].edges[2], true).score << endl;
	cout << Puzzle::matchEdges(pieces[13].edges[2], pieces[11].edges[3], true).score << endl;
	exit(0);
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

	// assemble
	myPuzzle.assemble(match_verbose);
	cout<<"Assembled" << endl;

	myPuzzle.print();
	myPuzzle.display(true);

	return 0;
}

double PuzzlePiece::scalingLength = 0;
double PuzzlePiece::avgBrightness = 0;

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

PuzzlePiece::PuzzlePiece(Mat m, int i) {
	img = m;
	number = i+1;
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
	double total_v = 0;
	for(Scalar c: backgroundColors) {
		if(c[0] < h_channel_min) h_channel_min = c[0];
		if(c[0] > h_channel_max) h_channel_max = c[0];
		if(c[1] < s_channel_min) s_channel_min = c[1];
		if(c[1] > s_channel_max) s_channel_max = c[1];
		if(c[2] < v_channel_min) v_channel_min = c[2];
		if(c[2] > v_channel_max) v_channel_max = c[2];
		total_v += c[2];
	}
	double h_channel_width = h_channel_max - h_channel_min;
	double s_channel_width = s_channel_max - s_channel_min;
	double v_channel_width = v_channel_max - v_channel_min;
	double v_avg = total_v / backgroundColors.size();

	// for light green background: (0.5, 1.5, 1.5)
	// for white background: (1, 5, 5) is ok not great
	double hueBuffer = 2;  // fraction denominator of color range that is added to each end
	double satBuffer = 2;
	double valueBuffer = 2;
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
	Mat close_kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(color_mask, color_mask, MORPH_CLOSE, close_kernel);
	if(verbose) {
		imshow("grey", color_mask);
		waitKey(0);
	}

	// scale the brightnes level
	if(PuzzlePiece::avgBrightness == 0) {
		PuzzlePiece::avgBrightness = v_avg;
	}
	img = img * PuzzlePiece::avgBrightness / v_avg;

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
	double pi = atan(1)*4;
	Rect outlineBound = boundingRect(outline);
	Rect coinBound = boundingRect(coin);
	double outlineRadius = max(outlineBound.width, outlineBound.height) / 2;
	double coinRadius = max(coinBound.width, coinBound.height) / 2;
	double coinCircleArea = pi * pow(coinRadius, 2);
	double outlineCircleArea = pi * pow(outlineRadius, 2);
	// swap outline and circle if needed
	if(coinCircleArea / coinArea > outlineCircleArea / outlineArea) {
		vector<Point> temp = coin;
		coin = outline;
		outline = temp;
		coinRadius = outlineRadius;
	}

	// scale everything up to regular size
	// also include the scale factor
	if(scalingLength == 0) {
		scalingLength = coinRadius;
	}
	cout << "debug: diameter: " << 2*coinRadius << endl;
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
		if(edges[i].isEdgeVar) { continue; }

		// rotate and translate the edges for easier comparison
		// translate all the edges to line up with the origin
		// (easier to line up edges from images w different dimensions. can translate again to display.)
		Point midPoint;
		if(i == 0) { midPoint = core.tl() + Point(core.width/2, 0); }
		if(i == 1) { midPoint = core.tl() + Point(core.width, core.height/2); }
		if(i == 2) { midPoint = core.tl() + Point(core.width/2, core.height); }
		if(i == 3) { midPoint = core.tl() + Point(0, core.height/2); }
		midpoints[i] = midPoint;
		for(Point &p: edges[i].edge) {
			p -= midPoint;
		}
	}

	// right edge: rotate by flipping across y = -x, then across y = 0 (origin is upper left)
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
	// left edge: rotate by flipping across y = x then across y = 0 (origin is upper left)
	if(!edges[3].isEdgeVar) {
		for(Point &p: edges[3].edge) {
			double temp = p.y;
			p.y = p.x;
			p.x = -temp;
		}
	}

	// create raster images of the edges
	int dotRadius = 10;  // 10 looked visually good
	for(int i = 0; i < 4; i++) {
		if(edges[i].isEdgeVar) continue;
		Rect edgeBound = boundingRect(edges[i].edge);
		Mat edgeImg = Mat::zeros(edgeBound.height + 2*dotRadius, edgeBound.width + 2*dotRadius, CV_8UC1);
		edges[i].rasterShift = - Point(edgeBound.x, edgeBound.y) + Point(dotRadius, dotRadius);
		for(Point p: edges[i].edge) {
			Point circleLoc = p + edges[i].rasterShift;
			circle(edgeImg, circleLoc, dotRadius, 255, -1);
		}
		edges[i].edgeImg = edgeImg;  // same name causes any problems?

		// rotated image (there must be a faster way)
		Mat rotatedEdgeImg = Mat::zeros(edgeImg.size(), edgeImg.type());
		for(int row = 0; row < edgeImg.rows; row++) {
			for(int col = 0; col < edgeImg.cols; col++) {
				rotatedEdgeImg.at<uchar>(row, col) = edgeImg.at<uchar>(edgeImg.rows-row-1, edgeImg.cols-col-1);
			}
		}
		edges[i].edgeImg180 = rotatedEdgeImg;
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

			if(!edges[0].isEdgeVar) {
				imshow("grey", edges[0].edgeImg);
				waitKey(0);
			}
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

Point Puzzle::calculateShift(Mat e1, Mat e2, int e1_row_max, int e2_row_max, int e1_col_min, int e2_col_min, int minWidth) {

	Point shift;

	if(e1.rows <= e2.rows) {
		shift.y = -(e2.rows - e2_row_max);
	} else {
		shift.y = e1.rows - e1_row_max;
	}

	if(e1.cols <= e2.cols) {
		shift.x = e2.cols/2 - (e2_col_min + minWidth/2);
	} else {
		shift.x = (e1_col_min + minWidth/2) - e1.cols/2;
	}

	return shift;
}

// close to 0 is a good match
// "bestShift" is amount 2nd edge needed to be shifted
PieceMatch Puzzle::matchEdges(EdgeOfPiece firstEdge, EdgeOfPiece other, bool verbose) {

	int pixelShift = 5;

	int minHeight = min(firstEdge.edgeImg.rows, other.edgeImg.rows);
	int maxHeight = max(firstEdge.edgeImg.rows, other.edgeImg.rows);
	int h_intervals = (maxHeight - minHeight) / pixelShift;
	int minWidth = min(firstEdge.edgeImg.cols, other.edgeImg.cols);
	int windowWidth = minWidth * 9 / 10;
	int maxWidth = max(firstEdge.edgeImg.cols, other.edgeImg.cols);
	int w_intervals = (maxWidth - windowWidth) / pixelShift;

	double minScore;
	int bestTheta;
	Point bestShift;
	bool firstScore = true;
	Mat best_e1; // for display only
	Mat best_e2; // for display only

	for(int theta_deg = -6; theta_deg <= 6; theta_deg+=3) {

		double theta = theta_deg * 3.14 / 180;

		Mat rotEdgeImg;
		Point rotationCenter = Point(firstEdge.edgeImg.cols/2, firstEdge.edgeImg.rows/2);
		Mat rot_t = getRotationMatrix2D(rotationCenter, theta, 1);
		warpAffine(other.edgeImg180, rotEdgeImg, rot_t, other.edgeImg180.size(), INTER_LINEAR, BORDER_CONSTANT, 0);

		for(int h = 0; h < h_intervals + 1; h++) {
			for(int w = 0; w < w_intervals + 1; w++) {
				int e1_col_min;
				int e1_col_max;
				int e2_col_min;
				int e2_col_max;
				int e1_row_min;
				int e1_row_max;
				int e2_row_min;
				int e2_row_max;

				if(firstEdge.edgeImg.rows == minHeight) {
					e1_row_min = 0;
					e1_row_max = minHeight;
					e2_row_min = h * pixelShift;
					e2_row_max = h * pixelShift + minHeight;
				} else {
					e2_row_min = 0;
					e2_row_max = minHeight;
					e1_row_min = h * pixelShift;
					e1_row_max = h * pixelShift + minHeight;
				}

				if(firstEdge.edgeImg.cols == minWidth) {
					e1_col_min = 0;
					e1_col_max = windowWidth;
					e2_col_min = w * pixelShift;
					e2_col_max = w * pixelShift + windowWidth;
				} else {
					e2_col_min = 0;
					e2_col_max = windowWidth;
					e1_col_min = w * pixelShift;
					e1_col_max = w * pixelShift + windowWidth;
				}

				// use rotated edge for e2
				Mat e1 = firstEdge.edgeImg.rowRange(e1_row_min, e1_row_max).colRange(e1_col_min, e1_col_max);
				Mat e2 = rotEdgeImg.rowRange(e2_row_min, e2_row_max).colRange(e2_col_min, e2_col_max);

				double score = edgeComparisonScore(e1, e2);
				if(firstScore) {
					firstScore = false;
					minScore = score;
					bestTheta = theta;
					bestShift = Puzzle::calculateShift(e1, e2, e1_row_max, e2_row_max, e1_col_min, e2_col_min, minWidth) + firstEdge.rasterShift - other.rasterShift;
					best_e1 = e1;  // for display
					best_e2 = e2;
				} else if(score < minScore) {
					minScore = score;
					bestTheta = theta;
					bestShift = Puzzle::calculateShift(e1, e2, e1_row_max, e2_row_max, e1_col_min, e2_col_min, minWidth) + firstEdge.rasterShift - other.rasterShift;
					best_e1 = e1;  // for display
					best_e2 = e2;
				}
			}
		}
	}

	if(verbose) {
		Mat bothEdges = Mat::zeros(best_e1.size(), CV_8UC3);
		Mat channels[3] = {best_e1, Mat::zeros(best_e1.size(), CV_8UC1), best_e2};
		merge(channels, 3, bothEdges);

		namedWindow("edgeMatch");
		imshow("edgeMatch", bothEdges);
		waitKey(0);
		destroyWindow("edgeMatch");
	}

	PieceMatch bestMatch;
	bestMatch.score = minScore;
	bestMatch.theta = bestTheta;
	bestMatch.shift = bestShift;
	return bestMatch;

	// ISSUE: these print blanks
	// cout << e1.at<uchar>(0, 0) << endl;
	// cout << e2.at<uchar>(e2.rows-1, e2.cols-1) << endl;

	// prep by rotating one edge (by flipping twice)
	vector<Point> flippedEdge = vector<Point>(other.edge.size());
	for(int i = 0; i < other.edge.size(); i++) {
		flippedEdge[i].x = -other.edge[i].x;
		flippedEdge[i].y = -other.edge[i].y;
	}

	// show the edges being compared
	Mat blank = Mat::zeros(3000, 3000, CV_8UC3);
	vector<vector<Point>> twoEdges = {firstEdge.edge, flippedEdge};
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

	// return matchShapes(edge, other.edge, CONTOURS_MATCH_I3, 0);
}

// search through edges to find a match
PieceMatch Puzzle::match(PuzzlePiece *piece, int edgeIndex, bool verbose) {

	PieceMatch bestMatch;
	bool firstMatch = true;
	double bestMatchScore;

	if(piece->edges[edgeIndex].isEdgeVar) {
		cout << "ERROR: calling match() on an edge piece" << endl;
		bestMatch.piece = nullptr;
		return bestMatch;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdgeVar) continue; // skip if it's an edge
			PieceMatch currentMatch = Puzzle::matchEdges(piece->edges[edgeIndex], pieces[i].edges[j], verbose);

			cout << "Piece " << piece->number << " scores " << currentMatch.score << " against index " << j << " of piece " << i+1 << endl;

			if(firstMatch) {
				bestMatchScore = currentMatch.score;
				bestMatch.piece = &pieces[i];
				bestMatch.edgeIndex = j;
				firstMatch = false;
			}
			else if(currentMatch.score < bestMatchScore) { // low score is best
				bestMatchScore = currentMatch.score;
				bestMatch.piece = &pieces[i];
				bestMatch.edgeIndex = j;
			}
		}
	}

	if(firstMatch) {
		cout << "ERROR: remaining pieces are edges only." << endl;
		bestMatch.piece = nullptr;
		return bestMatch;
	}

	cout << "Piece " << piece->number << " matches edge " << bestMatch.edgeIndex << " of piece " << bestMatch.piece->number << endl;

	return bestMatch;
}

PieceMatch Puzzle::match2(PuzzlePiece *leftPiece, int edgeIndexOfLeft, PuzzlePiece *upPiece, int edgeIndexOfUp, bool verbose) {

	PieceMatch bestMatchLeft;
	PieceMatch bestMatchUp;
	bool firstMatch = true;
	double bestMatchScore;

	if(leftPiece->edges[edgeIndexOfLeft].isEdgeVar || upPiece->edges[edgeIndexOfUp].isEdgeVar) {
		cout << "ERROR: calling match() on an edge piece" << endl;
		bestMatchLeft.piece = nullptr;
		return bestMatchLeft;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdgeVar || pieces[i].edges[(j+1)%4].isEdgeVar) continue; // skip if either connection is an edge
			PieceMatch matchLeft = Puzzle::matchEdges(leftPiece->edges[edgeIndexOfLeft], pieces[i].edges[j], verbose);
			PieceMatch matchUp = Puzzle::matchEdges(upPiece->edges[edgeIndexOfUp], pieces[i].edges[(j+1)%4], verbose);
			double score = matchLeft.score + matchUp.score;
			cout << "Pieces " << leftPiece->number << " and " << upPiece->number <<
					" score " << score << " against Piece " << pieces[i].number <<
					" with right index " << (j+2)%4  << endl;

			if(firstMatch) {
				firstMatch = false;
				bestMatchScore = score;
				bestMatchLeft = matchLeft;
				bestMatchUp = matchUp;
				bestMatchLeft.piece = &pieces[i];
				bestMatchLeft.edgeIndex = (j+2)%4;  // right index
			}
			else if(score < bestMatchScore) { // low score is best
				bestMatchScore = score;
				bestMatchLeft = matchLeft;
				bestMatchUp = matchUp;
				bestMatchLeft.piece = &pieces[i];
				bestMatchLeft.edgeIndex = (j+2)%4;  // right index
			}
		}
	}

	if(firstMatch) {
		cout << "ERROR: remaining pieces are edges only." << endl;
		bestMatchLeft.piece = nullptr;
		return bestMatchLeft;
	}

	// average corrections from up piece and left piece
	// rotate the correction from upMatch
	bestMatchUp.shift = Point(bestMatchUp.shift.y, -bestMatchUp.shift.x);  // counter clockwise
	bestMatchLeft.shiftLeft = bestMatchLeft.shift;
	bestMatchLeft.shiftUp = bestMatchUp.shift;
	bestMatchLeft.thetaLeft = bestMatchLeft.theta;
	bestMatchLeft.thetaUp = bestMatchUp.theta;
	bestMatchLeft.score = bestMatchScore;

	cout << "Pieces " << leftPiece->number << " and " << upPiece->number << " match Piece "
			<< bestMatchLeft.piece->number << " with right index " << bestMatchLeft.edgeIndex  << endl;

	return bestMatchLeft;
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
// may be able to deprecate this; only used once to detemine size of final puzzle
double PuzzlePiece::width() {
	if(rotationAngle() == 0 || rotationAngle() == 180) return core.width;
	return core.height;
}

double PuzzlePiece::height() {
	if(rotationAngle() == 0 || rotationAngle() == 180) return core.height;
	return core.width;
}

Point PuzzlePiece::center() {
	double x_center = core.tl().x + core.width / 2;
	double y_center = core.tl().y + core.height / 2;
	return Point(x_center, y_center);
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

// image, outline, core, edges
void PuzzlePiece::scale(double factor) {
	resize(img, img, Size(img.size[1] * factor, img.size[0] * factor));  // does this resize the Mat?
	for(Point &p: outline) {
		p.x *= factor;
		p.y *= factor; // can multiply a Point?
	}
	core = Rect(core.x * factor, core.y * factor, core.width * factor, core.height * factor);
	// skip edges
}

// image, outline, core, edges
void PuzzlePiece::shift(Point s, Size newSize) {
	double t_values[] = {1, 0, (double)s.x,
										0, 1, (double)s.y};
	Mat t_shift = Mat(2, 3, DataType<double>::type, t_values); // not sure about that data type
	warpAffine(img, img, t_shift, newSize);  // may want to preserve the original image...
	for(Point &p: outline) { p += s; }
	core = Rect(core.x + s.x, core.y + s.y, core.width, core.height);
	// skip edges
}

// allow for other angles of rotation (esp. for core)
void PuzzlePiece::rotate(Point rotationCenter, double theta) {
	Mat t = getRotationMatrix2D(rotationCenter, theta, 1);
	warpAffine(img, img, t, img.size());

	for(Point &p: outline) {
		double temp_x = p.x;
		p.x = t.at<double>(0, 0) * p.x + t.at<double>(0, 1) * p.y + t.at<double>(0, 2);
		p.y = t.at<double>(1, 0) * temp_x + t.at<double>(1, 1) * p.y + t.at<double>(1, 2);
	}

	// rotate core, assuming rotationCenter is center of piece
	if(theta == 90 || theta == 270) {
		core = Rect(core.tl().x + core.width/2 - core.height/2, core.tl().y + core.height/2 - core.width/2, core.height, core.width);
	} // else core stays the same
}

Puzzle::Puzzle(int _numPieces, PuzzlePiece _pieces[]) {
	numPieces = _numPieces;
	pieces = _pieces;
}

void Puzzle::process(bool verbose) {
	for(int i = 0; i < numPieces; i++) {
		pieces[i].process(verbose);
	}
}

void Puzzle::assemble(bool verbose) {

	// find a corner
	int firstCornerIdx = 0;
	while(firstCornerIdx < numPieces && !(pieces[firstCornerIdx]).isCorner()) {
		firstCornerIdx++;
	}
	if(firstCornerIdx >= numPieces) {
		cout << "ERROR: no corners found" << endl;
		return;
	}
	cout << "First corner at index " << firstCornerIdx << endl;

	completedPuzzle.push_back(vector<PuzzlePiece*>());
	completedPuzzle[0].push_back(&pieces[firstCornerIdx]);
	pieces[firstCornerIdx].isConnected = true;
	completedPuzzle[0][0]->rightIndex = completedPuzzle[0][0]->orientRoot();  // set orientation of top left corner
	cout << "Root is piece " << completedPuzzle[0][0]->number << endl;

	cout << "Start matching." << endl;

	// construct top edge
	cout << "Constructing top edge." << endl;
	PuzzlePiece *cursor = completedPuzzle[0][0];
	while(!cursor->edges[cursor->rightIndex].isEdgeVar && completedPuzzle[0].size() < numPieces) {
		PieceMatch matchingPiece = match(cursor, cursor->rightIndex, verbose);
		if(matchingPiece.piece == nullptr) {
			cout << "ERROR: no valid matches found" << endl;
			return;
		}

		cursor = matchingPiece.piece;
		completedPuzzle[0].push_back(cursor);
		cout << "Match: piece " << cursor->number << endl;
		cursor->isConnected = true;
		cursor->rightIndex = PuzzlePiece::oppIndex(matchingPiece.edgeIndex);
		// rotate clockwise 90deg (bc 180 deg then counter clock 90)
		cursor->correctionShiftLeft = Point(-matchingPiece.shift.y, matchingPiece.shift.x);
		cursor->correctionThetaLeft = matchingPiece.theta;
	}
	columns = completedPuzzle[0].size();
	// todo: check if all the pieces have been used. if so, is rightmost piece an edge?

	// calculate number of rows
	if(numPieces % columns != 0) {
		cout << "ERROR: " << columns << " columns found. Invalid for puzzle with " << numPieces << " pieces." << endl;
	} else {
		rows = numPieces / columns; // check division rules
		cout << rows << "rows" << endl;
	}

	// construct left edge
	cout << "Constructing left edge." << endl;
	cursor = completedPuzzle[0][0];  // reset cursor
	for(int i = 1; i < rows; i++) {
		if(cursor->edges[cursor->downIndex()].isEdgeVar) {
			cout << "ERROR: unexpected edge encountered" << endl;
			return;
		}

		PieceMatch matchingPiece = match(cursor, cursor->downIndex(), verbose);
		if(matchingPiece.piece == nullptr) {
			cout << "ERROR: no valid matches found" << endl;
			return;
		}

		cursor = matchingPiece.piece;
		completedPuzzle.push_back(vector<PuzzlePiece*>());
		completedPuzzle[i].push_back(cursor);  // combine w previous line?
		cout << "Match: piece " << cursor->number << endl;
		cursor->isConnected = true;
		cursor->rightIndex = PuzzlePiece::nextIndex(matchingPiece.edgeIndex);
		cursor->correctionShiftUp = -matchingPiece.shift;  // 180deg rotation (bc compare upside down)
		cursor->correctionThetaUp = matchingPiece.theta;
	}

	// fill in the rest:
	for(int i = 1; i < rows; i++) {
		for(int j = 1; j < columns; j++) {
			PuzzlePiece *leftPiece = completedPuzzle[i][j-1];
			PuzzlePiece *upPiece = completedPuzzle[i-1][j];
			PieceMatch matchingPiece = match2(leftPiece, leftPiece->rightIndex, upPiece, upPiece->downIndex(), verbose);
			if(matchingPiece.piece == nullptr) {
				cout << "ERROR: no match found" << endl;
				return;
			}
			cursor=matchingPiece.piece;
			completedPuzzle[i].push_back(cursor);
			cursor->isConnected = true;
			cursor->rightIndex = matchingPiece.edgeIndex;
			// todo: calculate this correction based on upper AND left pieces
			cursor->correctionShift = Point(-matchingPiece.shift.y, matchingPiece.shift.x);  // rotate clockwise 90deg
			cursor->correctionThetaLeft = matchingPiece.thetaLeft;
			cursor->correctionThetaUp = matchingPiece.thetaUp;
		}
	}
	// todo: check for edges in the middle of the puzzle

	cout << "Puzzle completed!" << endl;

	// todo: better evaluation of whether the puzzle is actually completed
}

// print piece numbers in solved configuration
void Puzzle::print() {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
			completedPuzzle[i][j]->print();
		}
		cout << endl;
	}
}

void Puzzle::display(bool verbose, bool checkRotation) {

	namedWindow("temp");

	// figure out the size needed (come up with better way)
	double width = completedPuzzle[0][0]->width();
	double height = completedPuzzle[0][0]->height();
	double edgeLength = max(columns * width, rows * height); // leave room for rotating at the end
	Mat completedPuzzleImg = Mat::zeros(edgeLength, edgeLength, completedPuzzle[0][0]->img.type());
	cout << "completed puzzle size: " << completedPuzzleImg.size() << endl;

	// loop through the pieces and copy to completed image
	for(int col = 0; col < columns; col++) {
		for(int row = 0; row < rows; row++) {

 			// todo: the rotation moves the top left corner if pieces are not square. need to account for that.
			// use the new top-left corner in the calculations instead.

			// if attaching left: rotate the left edge centroid. then line it up with the (rotated) centroid of the piece to the left.

			PuzzlePiece *cursor = completedPuzzle[row][col];
			PuzzlePiece *leftNeighbor;
			PuzzlePiece *upNeighbor;

			// detemine rotation and shift corrections
			Point2f shift; // about converting int to double if I just do Point
			double correctionTheta = 0;

			if(row == 0 && col == 0) {  // top left corner
				shift = -(cursor->core.tl());
				correctionTheta = 0;  // todo: correct rotation of the whole puzzle at the end
			}  else if(col == 0) {  // left edge
				PuzzlePiece *upNeighbor = completedPuzzle[row-1][col];
				Point upNeighborPoint = upNeighbor->core.br() + Point(-upNeighbor->core.width / 2);
				Point upPoint = cursor->core.tl() + Point(cursor->core.width / 2, 0);
				shift = upNeighborPoint - upPoint + cursor->correctionShiftUp;
				correctionTheta = cursor->correctionTheta - upNeighbor->actualAdditionalRotation;
			} else if(row == 0) {  // top edge
				PuzzlePiece *leftNeighbor = completedPuzzle[row][col-1];
				Point leftNeighborPoint = leftNeighbor->core.br() + Point(0, -leftNeighbor->core.height / 2);
				Point leftPoint = cursor->core.tl() + Point(0, cursor->core.height/2);
				shift = leftNeighborPoint - leftPoint + cursor->correctionShiftLeft;
				correctionTheta = cursor->correctionTheta - leftNeighbor->actualAdditionalRotation;
			} else {  // most pieces
				PuzzlePiece *upNeighbor = completedPuzzle[row-1][col];
				Point upNeighborPoint = upNeighbor->core.br() + Point(-upNeighbor->core.width / 2);
				Point upPoint = cursor->core.tl() + Point(cursor->core.width / 2, 0);
				Point shiftUp = upNeighborPoint - upPoint + cursor->correctionShiftLeft;
				PuzzlePiece *leftNeighbor = completedPuzzle[row][col-1];
				Point leftNeighborPoint = leftNeighbor->core.br() + Point(0, -leftNeighbor->core.height / 2);
				Point leftPoint = cursor->core.tl() + Point(0, cursor->core.height/2);
				Point shiftLeft = leftNeighborPoint - leftPoint + cursor->correctionShiftUp;
				shift = (shiftUp + shiftLeft) / 2;
				correctionTheta = (cursor->correctionThetaLeft - leftNeighbor->actualAdditionalRotation + cursor->correctionThetaUp - upNeighbor->actualAdditionalRotation) / 2;
			}

			// rotate about the center of the piece
			// rotate before scaling so the width / height are correct
			Point rotationCenter = Point(cursor->core.tl().x + cursor->core.width/2, cursor->core.tl().y + cursor->core.height/2);
			cursor->rotate(rotationCenter, cursor->rotationAngle());  // only 90 and 270 cause core to rotate
			cursor->rotate(rotationCenter, correctionTheta);
			cursor->actualAdditionalRotation = correctionTheta;
			if(verbose) {
				cout << "after just rotation:" << endl;
				imshow("temp", cursor->img);
				waitKey(0);
			}

			/*
			// scale the piece based on up and left neighbors
			double scaleFactor = 1;
			if(row == 0 && col == 0) { scaleFactor = 1; }
			else if(row == 0) {
				leftNeighbor = completedPuzzle[0][col-1];
				scaleFactor = (double)(leftNeighbor->core.height) / cursor->core.height;
			} else if(col == 0) {
				upNeighbor = completedPuzzle[row-1][0];
				scaleFactor = (double)(upNeighbor->core.width) / cursor->core.width;
			} else {
				leftNeighbor = completedPuzzle[row][0];  // all the way left
				upNeighbor = completedPuzzle[0][col];  // all the way up
				scaleFactor = (((double)(leftNeighbor->core.height) / cursor->core.height) + ((double)(upNeighbor->core.width) / cursor->core.width)) / 2;
			}
			cout << "Scale factor: " << scaleFactor << endl;
			cursor->scale(scaleFactor);

			if(verbose) {
				cout << "scaled image" << endl;
				imshow("temp", cursor->img);
				waitKey(0);
			}
			*/

			// now translate
			// should do in one step; I'm counting on no clipping after the rotation
			// note: could do this check implicitly by waiting until this while loop exits to the other one

			cursor->shift(shift, completedPuzzleImg.size());  // this makes cursor->img permanently much bigger, taking up a lot of RAM...
			if(verbose) {
				cout << "show completed transformation" << endl;
				imshow("temp", cursor->img);
				waitKey(0);
			}

			// copy the data within the piece outline to the final image
			Mat mask = Mat::zeros(cursor->img.size(), cursor->img.type());
			vector<vector<Point>> outlines = {cursor->outline};
			drawContours(mask, outlines, -1, Scalar(255, 255, 255), -1); // thickness=-1 fills in the contour
			if(verbose) {
				cout << "show mask" << endl;
				imshow("temp", mask);
				waitKey(0);
			}
			cursor->img.copyTo(completedPuzzleImg, mask);
			if(verbose) {
				cout << "show completed puzzle with new piece" << endl;
				imshow("temp", completedPuzzleImg);
				waitKey(0);
			}
		}
	}

	if(checkRotation) cout << "Enter degrees of clockwise rotation" << endl;

	// show completed puzzle
	imshow("temp", completedPuzzleImg);
	waitKey(0);

	if(checkRotation) {
		string rotationStr;
		cin >> rotationStr; // not sure how this interacts with waitKey()
		int fullPuzzleRotation = stoi(rotationStr); // error handling
		Point puzzleCenter = Point((columns * height) / 2, (rows * width) / 2);
		Mat t3 = getRotationMatrix2D(puzzleCenter, fullPuzzleRotation, 1);
		Mat rotatedPuzzle = Mat::zeros(completedPuzzleImg.size(), completedPuzzleImg.type()); // don't want any remnants after the rotation
		warpAffine(completedPuzzleImg, rotatedPuzzle, t3, rotatedPuzzle.size());
		imshow("temp", rotatedPuzzle);
		waitKey(0);
	}

	destroyWindow("temp");
}

// should do something to avoid pieces of other edges that are attached, etc.
// it's easy to match dark space. weight matching 1s and penalize non-matching 1s (xor).
// note: if edge1 is large it can have parts cut off when comparing to smaller edges.
//    and() is not affected but miss out on some xor() penalty. may be worth to fix.
double edgeComparisonScore(Mat edge1, Mat edge2) {

	Mat xor_mat;
	Mat and_mat;
	bitwise_xor(edge1, edge2, xor_mat);
	bitwise_and(edge1, edge2, and_mat);
	return (sum(xor_mat)[0] - 2 * sum(and_mat)[0]) / 255;

	// double score = 0;

//	for(int row = 0; row < edge1.rows; row++) {
//		for(int col = 0; col < edge1.cols; col++) {
//			// effectively rotating the 2nd image by flipping both coordinates
//			score += abs((int)edge1.at<uchar>(row, col) - (int)edge2.at<uchar>(edge2.rows-row-1, edge2.cols-col-1));
//		}
//	}
//	score /= (edge1.rows * edge1.cols);
//	return score;
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
