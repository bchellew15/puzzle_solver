#include <iostream>
#include <fstream>
#include <iomanip> // for setw(n)
#include <cmath>
#include <chrono>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/shape.hpp>
using namespace cv;

#include "PuzzleSolver2.h"

int main() {

	bool process_verbose = true;
	bool match_verbose = false;
	bool display_verbose = false;

	int numPieces = 12;
	cout << numPieces << " pieces" << endl;

	// load images
	Mat images[numPieces];
	PuzzlePiece pieces[numPieces];
	string dir = "/Users/blakechellew/Documents/Code/workspace/PuzzleSolver2/Pieces_12/";
	for( int i = 0; i < numPieces; i++) {
		string filename = dir + "Piece" + to_string(i+1) + ".jpeg";
		images[i] = imread(filename);
		pieces[i] = PuzzlePiece(images[i], i);
	}

	// create a Puzzle
	Puzzle myPuzzle = Puzzle(numPieces, pieces);

	chrono::time_point<chrono::steady_clock> start_time = chrono::steady_clock::now();
	myPuzzle.process(process_verbose);
	chrono::time_point<chrono::steady_clock> end_time = chrono::steady_clock::now();
	cout << "Processing time: " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << endl;

	// Test::displayEdgeMatches(myPuzzle);
	// Test::testAllEdgePairs(myPuzzle, true);
	// exit(0);

	// test puzzle: set edges
	pieces[1].edges[3].isEdge = false;
	pieces[10].isEdge = true;
	pieces[10].edges[0].isEdge = false;
	pieces[9].isEdge = true;
	pieces[9].edges[2].isEdge = false;
	pieces[3].isEdge = true;
	pieces[3].edges[0].isEdge = false;
	pieces[3].edges[1].isEdge = false;
	pieces[4].isEdge = true;
	pieces[4].edges[3].isEdge = false;
	pieces[2].edges[2].isEdge = false;

	// assemble
	start_time = chrono::steady_clock::now();
	myPuzzle.assemble(match_verbose);
	end_time = chrono::steady_clock::now();
	cout << "Assembly time: " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << endl;
	cout <<"Assembled" << endl;

	// display
	myPuzzle.print();
	start_time = chrono::steady_clock::now();
	myPuzzle.display(display_verbose);
	end_time = chrono::steady_clock::now();
	cout << "Display time: " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << endl;

	return 0;
}

int EdgeOfPiece::edgeHeightBuffer = 5;
double EdgeOfPiece::edgeShrinkFactor = 5;
int EdgeOfPiece::pixelShift = 5 / EdgeOfPiece::edgeShrinkFactor;
double PuzzlePiece::scalingLength = 0;
double PuzzlePiece::avgBrightness = 0;

pair<Mat, Point> EdgeOfPiece::rasterizeContour(vector<Point> contour, bool inverted) {

	Rect bound = boundingRect(contour);
	Mat img = Mat::zeros(bound.height / edgeShrinkFactor + 2 * edgeHeightBuffer, bound.width / edgeShrinkFactor + 1, CV_8UC1);
	Point shift = -Point(bound.x, bound.y) / edgeShrinkFactor + Point(0, edgeHeightBuffer);
	vector<Point> rasterLocs;
	for(Point p: contour) {
		rasterLocs.push_back(p / edgeShrinkFactor + shift);
	}
	if(inverted) {  // fill to the top corners
		rasterLocs.push_back(Point(img.cols, 0));
		rasterLocs.push_back(Point(0, 0));
	} else {  // fill to the bottom corners
		rasterLocs.push_back(Point(0, img.rows));
		rasterLocs.push_back(Point(img.cols, img.rows));
	}
	vector<vector<Point>> rasterLocsVec({rasterLocs});
	drawContours(img, rasterLocsVec, -1, 255, -1);  // thickness=-1 fills in the contour
	return pair<Mat, Point>(img, shift);
}

// fit line to decide if the edge is flat.
// if yes, calculate rotation and vertical shift required to line up the edge with border of puzzle.
// if no, create raster images of the edge.
void EdgeOfPiece::processEdge() {

	vector<Point>::const_iterator start_iter = edge.begin() + edge.size() * 2/10;
	vector<Point>::const_iterator end_iter = edge.begin() + edge.size() * 8/10;
	vector<Point> middleOfEdge(start_iter, end_iter);
	vector<double> fittedLine;
	fitLine(middleOfEdge, fittedLine, DIST_L2, 0.0, .01, .01);

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
		totalDist += abs(a*p.x + b*p.y + c) / denom;
	}
	double avgDist = totalDist / edge.size();

	if(avgDist < 10) {
		rotCorrection = atan(vy / vx) * 180 / 3.14;
		shiftCorrection = -(y0 - vy/vx*x0);
		cout << "x0, y0, vx, vy: " << x0 << " " << y0 << " " << vx << " " << vy << endl;
		cout << "processEdge shift correction: " << shiftCorrection << endl;
		cout << "processEdge rot correction: " << rotCorrection << endl;
		isEdge = true;
		return;
	}
	isEdge = false;

	// create raster images of the edge
	pair<Mat, Point> imgAndShift = rasterizeContour(edge, false);
	edgeImg = imgAndShift.first;
	rasterShift = imgAndShift.second;

	// raster images of rotated edges
	for(double theta = -4; theta <= 4; theta +=2) {
		rotEdgeImgAngles.push_back(theta);
		vector<Point> rotEdge;
		Mat rotMat = getRotationMatrix2D(Point(0, 0), 180 + theta, 1);
		for(const Point &p: edge) {
			rotEdge.push_back(rotatePoint(p, rotMat));
		}
		imgAndShift = rasterizeContour(rotEdge, true);
		rotEdgeImgs.push_back(imgAndShift.first);
		rotRasterShifts.push_back(imgAndShift.second);

//		namedWindow("temp");
//		imshow("temp", imgAndShift.first);
//		waitKey(0);
//		destroyWindow("temp");
	}
}

PuzzlePiece::PuzzlePiece() {}

PuzzlePiece::PuzzlePiece(Mat m, int i) {
	img = m;
	number = i+1;
}

// identify piece border and split into 4 edges
void PuzzlePiece::process(bool verbose) {

	Scalar blue(255, 0, 0);
	Scalar red(0, 0, 255);
	Scalar green(0, 255, 0);
	Scalar purple(128, 0, 128);
	vector<Scalar> colors = {blue, red, green, purple};

	// identify background color
	// todo: clean this up using minMaxLoc()
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	blur(img_hsv, img_hsv, Size(5, 5)); // TEST
	vector<Vec3b> backgroundColors;
	int cornerSize = 50;
	Mat topLeftCorner = img_hsv(Rect(0, 0, cornerSize, cornerSize));
	Mat topRightCorner = img_hsv(Rect(img.size().width-cornerSize, 0, cornerSize, cornerSize));
	Mat bottomLeftCorner = img_hsv(Rect(0, img.size().height-cornerSize, cornerSize, cornerSize));
	Mat bottomRightCorner = img_hsv(Rect(img.size().width-cornerSize, img.size().height-cornerSize, cornerSize, cornerSize));
	vector<Mat> cornerImgs = {topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner}; // this makes copies?
	int hChannelMin = topLeftCorner.at<Vec3b>(0, 0)[0];
	int hChannelMax = hChannelMin;
	int sChannelMin = topLeftCorner.at<Vec3b>(0, 0)[1];
	int sChannelMax = sChannelMin;
	int vChannelMin = topLeftCorner.at<Vec3b>(0, 0)[2];
	int vChannelMax = vChannelMin;
	for(Mat cornerImg: cornerImgs) {
		vector<Mat> channels;
		split(cornerImg, channels);
		double hMin; double hMax;
		double sMin; double sMax;
		double vMin; double vMax;
		minMaxLoc(channels[0], &hMin, &hMax);  // won't let hMin etc. be int
		minMaxLoc(channels[1], &sMin, &sMax);
		minMaxLoc(channels[2], &vMin, &vMax);

		if(hMin < hChannelMin) hChannelMin = hMin;
		if(hMax > hChannelMax) hChannelMax = hMax;
		if(sMin < sChannelMin) sChannelMin = sMin;
		if(sMax > sChannelMax) sChannelMax = sMax;
		if(vMin < vChannelMin) vChannelMin = sMin;
		if(vMax > vChannelMax) vChannelMax = sMax;
	}

	// create color mask
	Mat color_mask;
	double hueBuffer = 25;
	double satBuffer = 15;
	double valueBuffer = 255;
	Scalar colorLowerBound = Scalar(max(0.0, hChannelMin - hueBuffer), max(0.0, sChannelMin - satBuffer), max(0.0, vChannelMin - valueBuffer));
	Scalar colorUpperBound = Scalar(min(255.0, hChannelMax + hueBuffer), min(255.0, sChannelMax + satBuffer), min(255.0, vChannelMax + valueBuffer));
	inRange(img_hsv, colorLowerBound, colorUpperBound, color_mask);
	color_mask = 255 - color_mask;  // invert

	if(verbose) {
		cout << "HSV lower bounds: " << colorLowerBound << endl;
		cout << "HSV upper bounds: " << colorUpperBound << endl;
		namedWindow("mask");
		rectangle(color_mask, Rect(0, 0, cornerSize, cornerSize), 255, -1);
		rectangle(color_mask, Rect(0, color_mask.rows-cornerSize, cornerSize, cornerSize), 255, -1);
		rectangle(color_mask, Rect(color_mask.cols-cornerSize, 0, cornerSize, cornerSize), 255, -1);
		rectangle(color_mask, Rect(color_mask.cols-cornerSize, color_mask.rows-cornerSize, cornerSize, cornerSize), 255, -1);
		imshow("mask", color_mask);
		waitKey(0);
	}

	// morphological close
	Mat close_kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(color_mask, color_mask, MORPH_CLOSE, close_kernel);
	if(verbose) {
		imshow("mask", color_mask);
		waitKey(0);
		destroyWindow("mask");
	}

	/*
	// adjust the brightness level
	if(PuzzlePiece::avgBrightness == 0) {
		PuzzlePiece::avgBrightness = v_avg;
	}
	img = img * PuzzlePiece::avgBrightness / v_avg;
	*/

	// find contours
	vector<vector<Point>> contours;
	findContours(color_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if(contours.size() < 2) {  // piece and coin
		cout << "ERROR: no puzzle piece found" << endl;
		return;
	}

	Mat img_copy = img.clone();
	if(verbose) {
		cout << "display all contours" << endl;
		drawContours(img_copy, contours, -1, blue, 5);
		namedWindow("contours");
		imshow("contours", img_copy);
		waitKey(0);
	}

	// identify the biggest contours by area
	vector<Point> firstContour = contours[0];
	vector<Point> secondContour = contours[1];
	double maxSize = contourArea(firstContour);
	double secondMaxSize = contourArea(secondContour);
	for(int i = 1; i < contours.size(); i++) {
		double currentArea = contourArea(contours[i]);
		if(currentArea > maxSize) {
			secondMaxSize = maxSize;
			secondContour = firstContour;
			maxSize = currentArea;
			firstContour = contours[i];
		}
		else if(currentArea > secondMaxSize) {
			secondMaxSize = currentArea;
			secondContour = contours[i];
		}
	}

	// check which one is the circle
	double firstArea = contourArea(firstContour);
	double secondArea = contourArea(secondContour);
	double pi = atan(1)*4;
	Rect firstBox = boundingRect(firstContour);
	Rect secondBox = boundingRect(secondContour);
	double firstRadius = max(firstBox.width, firstBox.height) / 2;
	double secondRadius = max(secondBox.width, secondBox.height) / 2;
	double firstCircleArea = pi * pow(firstRadius, 2);
	double secondCircleArea = pi * pow(secondRadius, 2);
	// assign coin and outline based on circumscribed circle
	vector<Point> coin;
	double coinRadius;
	double outlineRadius;
	if(firstCircleArea / firstArea > secondCircleArea / secondArea) {
		coin = secondContour;
		coinRadius = secondRadius;
		outline = firstContour;
		outlineRadius = firstRadius;
		core = firstBox;
	} else {
		coin = firstContour;
		coinRadius = firstRadius;
		outline = secondContour;
		outlineRadius = secondRadius;
		core = secondBox;
	}

	// TEST: output contour points
	// draw just the outline
//	vector<vector<Point>> contourVec = {outline};
//	Mat justOutline = Mat::zeros(color_mask.size(), color_mask.type());
//	drawContours(justOutline, contourVec, 0, 255, -1);
//	close_kernel = getStructuringElement(MORPH_RECT, Size(10, 10));
//	morphologyEx(justOutline, justOutline, MORPH_CLOSE, close_kernel, Point(-1, 1), 5);
//	contourVec.clear();
////	findContours(justOutline, contourVec, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//	ofstream outlineFile("piece_outline.csv");
//	for(Point p: outline) {
//		outlineFile << p.x << "," << p.y << "\n";
//	}
//	outlineFile.close();
//	exit(0);

	// scale image and piece outline based on size of coin
	if(scalingLength == 0) {
		scalingLength = coinRadius;
	}
	double normalize_factor = scalingLength / coinRadius;
	cout << "Piece " << number << " scaling by " << normalize_factor << endl;
	scale(normalize_factor);

	if(verbose) {
		for(Point &p: coin) {
			p.x *= normalize_factor;
			p.y *= normalize_factor;
		}
		contours.clear();
		contours.push_back(outline);
		contours.push_back(coin);

		// show coin in red
		img_copy = img.clone();
		drawContours(img_copy, contours, 0, blue, 5);
		drawContours(img_copy, contours, 1, red, 5);
		imshow("contours", img_copy);
		waitKey(0);

		// show bounding box
		rectangle(img_copy, core.tl(), core.br(), blue, 5);
		imshow("contours", img_copy);
		waitKey(0);
	}

	// shrink the bounding box to get the "core" rectangle of the piece
	int scanWidth = max(core.width, core.height) / 16;
	int scanDepth = max(core.width, core.height) / 40;

	// scan from top
	vector<Point> scanLine;
	for(int i = 0; i < core.width / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(i * scanWidth, 0));
	}
	int topBuffer = scan(scanLine, Point(0, scanDepth), scanDepth, core.height);
	core = Rect(core.tl().x, core.tl().y + topBuffer, core.width, core.height - topBuffer);

	// scan from bottom
	scanLine.clear();
	for(int i = 0; i < core.width / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(i * scanWidth, core.height));
	}
	int bottomBuffer = scan(scanLine, Point(0, -scanDepth), scanDepth, core.height);
	core = Rect(core.tl().x, core.tl().y, core.width, core.height - bottomBuffer);

	// scan from left
	scanLine.clear();
	for(int i = 0; i < core.height / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(0, i * scanWidth));
	}
	int leftBuffer = scan(scanLine, Point(scanDepth, 0), scanDepth, core.width);
	core = Rect(core.tl().x + leftBuffer, core.tl().y, core.width - leftBuffer, core.height);

	// scan from right
	scanLine.clear();
	for(int i = 0; i < core.height / scanWidth; i++) {
		scanLine.push_back(core.tl() + Point(core.width, i * scanWidth));
	}
	int rightBuffer = scan(scanLine, Point(-scanDepth, 0), scanDepth, core.width);
	core = Rect(core.tl().x, core.tl().y, core.width - rightBuffer, core.height);

	if(verbose) {
		// show core
		rectangle(img_copy, core.tl(), core.br(), red, 5);
		imshow("contours", img_copy);
		waitKey(0);
	}

	// find closest point to each of corner of the core
	vector<string> keys = {"TL", "TR", "BL", "BR"};
	map<string, int> cornerIndices = {{"TL", 0}, {"TR", 0}, {"BL", 0}, {"BR", 0}};  // TL, TR, BL, BR
	map<string, double> cornerMinDists = {{"TL", norm(core.tl() - outline[0])},
															{"TR", norm(core.tl() + Point(core.width, 0) - outline[0])},
															{"BL", norm(core.br() + Point(-(core.width), 0) - outline[0])},
															{"BR", norm(core.br() - outline[0])}};
	for(int i = 0; i < outline.size(); i++) {
		map<string, double> cornerDists = {{"TL", norm(core.tl() - outline[i])},
															{"TR", norm(core.tl() + Point(core.width, 0) - outline[i])},
															{"BL", norm(core.br() + Point(-(core.width), 0) - outline[i])},
															{"BR", norm(core.br() - outline[i])}};
		for(string k: keys) {
			if (cornerDists[k] < cornerMinDists[k]) {
				cornerMinDists[k] = cornerDists[k];
				cornerIndices[k] = i;
			}
		}
	}
	edges[0].edge = constructEdge(cornerIndices["TR"], cornerIndices["TL"]);
	edges[1].edge = constructEdge(cornerIndices["BR"], cornerIndices["TR"]);
	edges[2].edge = constructEdge(cornerIndices["BL"], cornerIndices["BR"]);
	edges[3].edge = constructEdge(cornerIndices["TL"], cornerIndices["BL"]);

	if(verbose) {
		// reset the image and plot the edges
		vector<vector<Point>> edge_vector = {edges[0].edge, edges[1].edge, edges[2].edge, edges[3].edge};
		img_copy = img.clone();
		drawContours(img_copy, edge_vector, 0, blue, 5);
		drawContours(img_copy, edge_vector, 1, red, 5);
		drawContours(img_copy, edge_vector, 2, green, 5);
		drawContours(img_copy, edge_vector, 3, purple, 5);
		circle(img_copy, outline[cornerIndices["TL"]], 5, blue, 15);
		circle(img_copy, outline[cornerIndices["TR"]], 5, red, 15);
		circle(img_copy, outline[cornerIndices["BL"]], 5, green, 15);
		circle(img_copy, outline[cornerIndices["BR"]], 5, purple, 15);
		imshow("contours", img_copy);
		waitKey(0);
	}

	// rotate and translate the edges for easier processing and comparison
	// translate: line up core midpoints with the origin
	edges[0].midpoint = core.tl() + Point(core.width/2, 0);
	edges[1].midpoint = core.tl() + Point(core.width, core.height/2);
	edges[2].midpoint = core.tl() + Point(core.width/2, core.height);
	edges[3].midpoint = core.tl() + Point(0, core.height/2);
	for(int i = 0; i < 4; i++) {
		for(Point &p: edges[i].edge) {
			p -= edges[i].midpoint;
		}
	}
	// rotate edges
	for(Point &p: edges[1].edge) {  // right edge
		p = Point(p.y, -p.x);
	}
	for(Point &p: edges[2].edge) {  // bottom edge
		p = -p;
	}
	for(Point &p: edges[3].edge) {  // left edge
		p = Point(-p.y, p.x);
	}

	// process edges: check for flat edges and create raster edge images
	for(int i = 0; i < 4; i++) {
		edges[i].processEdge();
	}
	isEdge = edges[0].isEdge + edges[1].isEdge + edges[2].isEdge + edges[3].isEdge;

	if(verbose) {
			// show the shifted / rotated edges
			vector<vector<Point>> edge_vector = {edges[0].edge, edges[1].edge, edges[2].edge, edges[3].edge};
			for(int i = 0; i < 4; i++) {
				for(Point &p: edge_vector[i]) {
					p += Point(1000, 1000);  // shift for easier plotting
				}
			}
			img_copy = img.clone();
			for(int i = 0; i < 4; i++) {
				if(!edges[i].isEdge) drawContours(img_copy, edge_vector, i, colors[i], 5);
			}
			imshow("contours", img_copy);
			waitKey(0);

			// show one of the raster edge images
			if(!edges[0].isEdge) {
				imshow("contours", edges[0].edgeImg);
				waitKey(0);
			}
			destroyWindow("contours");
		}
}

int PuzzlePiece::scan(vector<Point> scanLine, Point increment, int scanDepth, int maxBuffer) {

	int startIndex = 0;
	int endIndex = 0;
	int numScans = 0;
	int buffer = 0;

	for(numScans = 0; (endIndex - startIndex) / (double)scanLine.size() < 0.5; numScans++) {
		buffer = numScans * scanDepth;
		if(buffer >= maxBuffer) {
			cout << "ERROR: unable to identify piece core" << endl;
			break;
		}

		for(startIndex = 0; startIndex < scanLine.size(); startIndex++) {
			if(pointPolygonTest(outline, scanLine[startIndex] + numScans * increment, false) == 1) {
				break;
			}
		}
		for(endIndex = scanLine.size() - 1; endIndex >= 0; endIndex--) {
			if(pointPolygonTest(outline, scanLine[endIndex] + numScans * increment, false) == 1) {
				break;
			}
		}
	}
	return buffer;
}

int PuzzlePiece::countEdges() {
	return edges[0].isEdge + edges[1].isEdge + edges[2].isEdge + edges[3].isEdge;
}

bool PuzzlePiece::isCorner() {
	return countEdges() >= 2;
}

// return right index of top left corner
// assumption: the piece has exactly 2 consecutive edges
int PuzzlePiece::rootRightIndex() {
	if(edges[0].isEdge) {
		if(edges[1].isEdge) {
			return 2;
		}
		else return 1;
	}
	else if(edges[1].isEdge) return 3;
	else return 0;
}

int PuzzlePiece::oppIndex(int index) {
	return (index + 2) % 4;
}

int PuzzlePiece::nextIndex(int index) {
	return (index + 1) % 4;
}

int PuzzlePiece::prevIndex(int index) {
	return (index +3) % 4;
}

int PuzzlePiece::downIndex() {
	return nextIndex(rightIndex);
}

int PuzzlePiece::upIndex() {
	return prevIndex(rightIndex);
}

int PuzzlePiece::leftIndex() {
	return oppIndex(rightIndex);
}

double EdgeOfPiece::edgeComparisonScore(Mat edge1, Mat edge2) {
	Mat nor_mat;
	Mat and_mat;
	bitwise_or(edge1, edge2, nor_mat);
	bitwise_not(nor_mat, nor_mat);
	bitwise_and(edge1, edge2, and_mat);
	return (sum(nor_mat)[0] + sum(and_mat)[0]) / 255;
}

// rename this
double EdgeOfPiece::edgeComparisonScore2(Mat edge, bool penalizeZeros) {
	if(penalizeZeros) {
		Mat not_mat;
		bitwise_not(edge, not_mat);
		return sum(not_mat)[0] / 255;
	} else {
		return sum(edge)[0] / 255;
	}
}

// lower score is better.
// "shift" and "theta" are corrections for edge 2.
// edge2 images are rotated ~180 degrees for comparison.
EdgeMatch EdgeOfPiece::matchEdges(EdgeOfPiece edge1, EdgeOfPiece edge2, bool verbose) {

	bool firstMatch = true;
	EdgeMatch bestMatch;

	for(int i = 0; i < edge2.rotEdgeImgAngles.size(); i++) {

		Mat rotEdgeImg = edge2.rotEdgeImgs[i];

		int minHeight = min(edge1.edgeImg.rows, rotEdgeImg.rows);
		int windowHeight = minHeight - 2 * edgeHeightBuffer;
		int maxHeight = max(edge1.edgeImg.rows, rotEdgeImg.rows);
		int h_intervals = (maxHeight - windowHeight) / pixelShift;
		int minWidth = min(edge1.edgeImg.cols, rotEdgeImg.cols);
		int windowWidth = minWidth * 8 / 10;
		int maxWidth = max(edge1.edgeImg.cols, rotEdgeImg.cols);
		int w_intervals = (maxWidth - windowWidth) / pixelShift;

		for(int h = 0; h < h_intervals + 1; h++) {
			for(int w = 0; w < w_intervals + 1; w++) {
				Range e1ColRange;
				Range e2ColRange;
				Range e1RowRange;
				Range e2RowRange;

				if(edge1.edgeImg.rows <= rotEdgeImg.rows) {
					e1RowRange = Range(edgeHeightBuffer, edgeHeightBuffer + windowHeight);
					e2RowRange = Range(h * pixelShift, h * pixelShift + windowHeight);
				} else {
					e2RowRange = Range(edgeHeightBuffer, edgeHeightBuffer + windowHeight);
					e1RowRange = Range(h * pixelShift, h * pixelShift + windowHeight);
				}
				if(edge1.edgeImg.cols <= rotEdgeImg.cols) {
					e1ColRange = Range(minWidth/10, minWidth/10 + windowWidth);
					e2ColRange = Range(w * pixelShift, w * pixelShift + windowWidth);
				} else {
					e2ColRange = Range(minWidth/10, minWidth/10 + windowWidth);
					e1ColRange = Range(w * pixelShift, w * pixelShift + windowWidth);
				}
				Mat e1 = edge1.edgeImg.colRange(e1ColRange);
				Mat e2 = rotEdgeImg.colRange(e2ColRange);

				double score = edgeComparisonScore(e1.rowRange(e1RowRange), e2.rowRange(e2RowRange));
				if(edge1.edgeImg.rows <= rotEdgeImg.rows) {
					// top cutoff, penalize dark pixels
					Range e2CutOffRows = Range(0, h * pixelShift);
					Mat e2CutOff = rotEdgeImg.rowRange(e2CutOffRows);
					if(e2CutOff.cols > 0) e2CutOff = e2CutOff.colRange(e2ColRange);
					score += edgeComparisonScore2(e2CutOff, true);

					// bottom cutoff, penalize white pixels
					e2CutOffRows = Range(h * pixelShift + windowHeight, rotEdgeImg.rows);
					e2CutOff = rotEdgeImg.rowRange(e2CutOffRows);
					if(e2CutOff.cols > 0) e2CutOff = e2CutOff.colRange(e2ColRange);
					score += edgeComparisonScore2(e2CutOff, false);
				} else {
					// top cutoff, penalize white pixels
					Range e1CutOffRows = Range(0, h * pixelShift);
					Mat e1CutOff = edge1.edgeImg.rowRange(e1CutOffRows);
					if(e1CutOff.cols > 0) e1CutOff = e1CutOff.colRange(e1ColRange);
					score += edgeComparisonScore2(e1CutOff, false);

					// bottom cutoff, penalize dark pixels
					e1CutOffRows = Range(h * pixelShift + windowHeight, edge1.edgeImg.rows);
					e1CutOff = edge1.edgeImg.rowRange(e1CutOffRows);
					if(e1CutOff.cols > 0) e1CutOff = e1CutOff.colRange(e1ColRange);
					score += edgeComparisonScore2(e1CutOff, true);
				}
				score *= maxWidth / (double)rotEdgeImg.cols;

				if(firstMatch || score < bestMatch.score) {
					if(firstMatch) firstMatch = false;

					bestMatch.score = score;
					bestMatch.theta = edge2.rotEdgeImgAngles[i];
					// (1) apply edge2 raster shift, but rotated about the center bc edge2 is rotated 180 degrees. -1 bc origin is center of TL pixel
					// (2) apply the shift calculated in this function
					// (3) apply reverse of edge1 raster shift
					Point shift = edge2.rotRasterShifts[i] + Point(e1ColRange.start - e2ColRange.start, e1RowRange.start - e2RowRange.start) - edge1.rasterShift;
					bestMatch.shift = shift * edgeShrinkFactor;

					// for display
					bestMatch.e1 = e1;
					bestMatch.e2 = e2;
					bestMatch.e1RowRange = e1RowRange;
					bestMatch.e2RowRange = e2RowRange;
					bestMatch.minHeight = minHeight;
					bestMatch.maxHeight = maxHeight;
					bestMatch.windowWidth = windowWidth;
					bestMatch.windowHeight = windowHeight;
				}
			}
		}
	}

	if(verbose) {
		cout << "best theta: " << bestMatch.theta << endl;

		Mat channel1 = Mat::zeros(bestMatch.maxHeight, bestMatch.windowWidth, CV_8UC1);
		Mat channel3 = Mat::zeros(bestMatch.maxHeight, bestMatch.windowWidth, CV_8UC1);
		// might be possible to do this without the if...
		if(edge1.edgeImg.rows <= bestMatch.e2.rows) {
			Rect edge1Box = Rect(0, bestMatch.e2RowRange.start, bestMatch.windowWidth, bestMatch.windowHeight);
			bestMatch.e1.rowRange(bestMatch.e1RowRange).copyTo(channel1(edge1Box));
			rectangle(channel1, Point(0, bestMatch.e2RowRange.end), Point(bestMatch.windowWidth, channel1.rows), 255, -1);
			bestMatch.e2.copyTo(channel3(Rect({}, bestMatch.e2.size())));
		} else {
			bestMatch.e1.copyTo(channel1(Rect({}, bestMatch.e1.size())));
			Rect edge2Box = Rect(0, bestMatch.e1RowRange.start, bestMatch.windowWidth, bestMatch.windowHeight);
			bestMatch.e2.rowRange(bestMatch.e2RowRange).copyTo(channel3(edge2Box));
			rectangle(channel3, Point(0, 0), Point(bestMatch.windowWidth, bestMatch.e1RowRange.start), 255, -1);
		}
		Mat bothEdges;
		Mat channels[3] = {channel1, Mat::zeros(channel1.size(), CV_8UC1), channel3};
		merge(channels, 3, bothEdges);

		cout << "score: " << bestMatch.score << endl;
		namedWindow("edgeMatch");
		imshow("edgeMatch", bothEdges);
		waitKey(0);
		destroyWindow("edgeMatch");
	}

	return bestMatch;
}

// search through remaining pieces and return best match for edge "edgeIndex" of piece "piece"
PieceMatch Puzzle::match(PuzzlePiece *piece, int edgeIndex, bool edgesOnly, bool verbose) {

	PieceMatch bestMatch;
	double bestMatchScore;
	bool firstMatch = true;

	if(piece->edges[edgeIndex].isEdge) {
		cout << "ERROR: calling match() on a flat edge" << endl;
		return bestMatch;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue;  // skip if already connected
		if(edgesOnly && !pieces[i].isEdge) continue;
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdge) continue;  // skip if it's an edge

			EdgeMatch currentMatch = EdgeOfPiece::matchEdges(piece->edges[edgeIndex], pieces[i].edges[j], verbose);
			cout << "Piece " << piece->number << " scores " << currentMatch.score << " against index " << j << " of piece " << i+1 << endl;

			if(firstMatch || currentMatch.score < bestMatchScore) {  // low score is best
				if(firstMatch) firstMatch = false;
				bestMatchScore = currentMatch.score;
				bestMatch.match = currentMatch;
				bestMatch.piece = &pieces[i];
				bestMatch.edgeIndex = j;
			}
		}
	}

	if(bestMatch.piece == nullptr) {
		cout << "ERROR: remaining pieces are edges only." << endl;
		return bestMatch;
	}

	cout << "Piece " << piece->number << " matches piece " << bestMatch.piece->number << ", index " << bestMatch.edgeIndex << endl;
	return bestMatch;
}

// search through remaining pieces and return the piece that fits best in the spot with neighbors "leftPiece" and "upPiece"
vector<PieceMatch> Puzzle::match2(PuzzlePiece *leftPiece, int edgeIndexOfLeft, PuzzlePiece *upPiece, int edgeIndexOfUp, bool noEdges, bool verbose) {

	PieceMatch bestMatchLeft;
	PieceMatch bestMatchUp;
	double bestMatchScore;
	bool firstMatch = true;

	if(leftPiece->edges[edgeIndexOfLeft].isEdge || upPiece->edges[edgeIndexOfUp].isEdge) {
		cout << "ERROR: calling match() on an edge piece" << endl;
		vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
		return matches;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue;  // skip if already connected
		if(noEdges && pieces[i].isEdge) continue;
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdge || pieces[i].edges[(j+1)%4].isEdge) continue;  // skip if either connection is an edge

			EdgeMatch edgeMatchLeft = EdgeOfPiece::matchEdges(leftPiece->edges[edgeIndexOfLeft], pieces[i].edges[j], verbose);
			EdgeMatch edgeMatchUp = EdgeOfPiece::matchEdges(upPiece->edges[edgeIndexOfUp], pieces[i].edges[(j+1)%4], verbose);
			double score = edgeMatchLeft.score + edgeMatchUp.score;
			cout << "Pieces " << leftPiece->number << " and " << upPiece->number << " score " << score <<
					" against Piece " << pieces[i].number << " with right index " << (j+2)%4  << endl;

			if(firstMatch || score < bestMatchScore) {  // low score is best
				if(firstMatch) firstMatch = false;
				bestMatchScore = score;
				bestMatchLeft.match = edgeMatchLeft;
				bestMatchUp.match = edgeMatchUp;
				bestMatchLeft.piece = &pieces[i];
				bestMatchUp.piece = &pieces[i];
				bestMatchLeft.edgeIndex = j;
				bestMatchUp.edgeIndex = j;
			}
		}
	}

	if(bestMatchLeft.piece == nullptr) {
		cout << "ERROR: remaining pieces have no adjacent connections." << endl;
		vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
		return matches;
	}

	cout << "Pieces " << leftPiece->number << " and " << upPiece->number << " match Piece "
			<< bestMatchLeft.piece->number << " with right index " << bestMatchLeft.edgeIndex  << endl;
	vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
	return matches;
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

Point PuzzlePiece::center() {
	return (core.tl() + core.br()) / 2;
}

vector<Point> PuzzlePiece::constructEdge(int firstIdx, int secondIdx) {
	if(secondIdx > firstIdx) {
		vector<Point> edge(outline.begin() + firstIdx, outline.begin() + secondIdx + 1);
		return edge;
	} else {
		vector<Point> edge(outline.begin() + firstIdx, outline.end());
		edge.insert(edge.end(), outline.begin(), outline.begin() + secondIdx + 1);
		return edge;
	}
}

// image, outline, core, edges
void PuzzlePiece::scale(double factor) {
	resize(img, img, Size(img.size[1] * factor, img.size[0] * factor));
	for(Point &p: outline) { p *= factor; }
	core = Rect(core.x * factor, core.y * factor, core.width * factor, core.height * factor);
	for(EdgeOfPiece &e: edges) { e.midpoint *= factor; }
}

// shift midpoints only
void PuzzlePiece::shift(Point s) {
	// double t_values[] = {1, 0, (double)s.x,
	// 									0, 1, (double)s.y};
	// Mat t_shift = Mat(2, 3, DataType<double>::type, t_values);
	// warpAffine(img, img, t_shift, newSize);
	// for(Point &p: outline) { p += s; }
	core = Rect(core.x + s.x, core.y + s.y, core.width, core.height);
	for(EdgeOfPiece &e: edges) { e.midpoint += s; }
}

// different behavior for theta = 90, 180, 270
void PuzzlePiece::rotate(Point rotationCenter, double theta) {
	Mat t = getRotationMatrix2D(rotationCenter, theta, 1);
	warpAffine(img, img, t, img.size());
	for(Point &p: outline) { p = rotatePoint(p, t); }
	for(EdgeOfPiece &e: edges) { e.midpoint = rotatePoint(e.midpoint, t); }

	// rotate core and midpoints for big rotations
	if(theta == 90) {
		Point core_tr = core.tl() + Point(core.width, 0);
		core = Rect(rotatePoint(core_tr, t), Size(core.height, core.width));
		cycleMidpoints();
	} else if(theta == 180) {  // core stays the same
		core = Rect(rotatePoint(core.br(), t), Size(core.width, core.height));
		cycleMidpoints(); cycleMidpoints();
	} else if(theta == 270) {
		Point core_bl = core.br() + Point(-core.width, 0);
		core = Rect(rotatePoint(core_bl, t), Size(core.height, core.width));
		cycleMidpoints(); cycleMidpoints(); cycleMidpoints();
	}

	// only for small corrections, and assuming rotation about center of piece
	else if(theta < 90) {
		// rotate edge corrections
		for(EdgeOfPiece &e: edges) {
			if(e.isEdge) {
				e.shiftCorrection -= norm(center() - e.midpoint) * (1 - cos(theta * 3.14 / 180));
			}
		}
	}
}

void PuzzlePiece::cycleMidpoints() {
	Point temp = edges[0].midpoint;
	edges[0].midpoint = edges[1].midpoint;
	edges[1].midpoint = edges[2].midpoint;
	edges[2].midpoint = edges[3].midpoint;
	edges[3].midpoint = temp;
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
	int firstCornerIdx;
	for(firstCornerIdx = 0; firstCornerIdx < numPieces; firstCornerIdx++) {
		if(pieces[firstCornerIdx].isCorner()) break;
	}
	if(firstCornerIdx >= numPieces) {
		cout << "ERROR: no corners found" << endl;
		return;
	}
	PuzzlePiece *cursor = &pieces[firstCornerIdx];
	cout << "First corner is piece number " << cursor->number << endl;

	completedPuzzle.push_back(vector<PuzzlePiece*>());
	completedPuzzle[0].push_back(cursor);
	cursor->isConnected = true;
	cursor->rightIndex = cursor->rootRightIndex();  // set orientation of top left corner

	// construct top edge
	cout << "Constructing top edge" << endl;
	while(!cursor->edges[cursor->rightIndex].isEdge && completedPuzzle[0].size() < numPieces) {

		PieceMatch matchingPiece = match(cursor, cursor->rightIndex, true, verbose);
		cursor = matchingPiece.piece;
		if(cursor == nullptr) {
			cout << "ERROR: no valid matches found" << endl;
			return;
		}

		cout << "Match: piece " << cursor->number << endl;
		completedPuzzle[0].push_back(cursor);
		cursor->isConnected = true;
		cursor->rightIndex = PuzzlePiece::oppIndex(matchingPiece.edgeIndex);
		// shiftLeft: rotate clockwise 90 degrees bc matchingPiece is above while matching
		cursor->correctionShiftLeft = Point(-matchingPiece.match.shift.y, matchingPiece.match.shift.x);
		cursor->correctionAngleLeft = matchingPiece.match.theta;
	}
	columns = completedPuzzle[0].size();

	// calculate number of rows
	if(numPieces % columns != 0) {
		cout << "ERROR: " << columns << " columns found. Invalid for puzzle with " << numPieces << " pieces." << endl;
	} else {
		rows = numPieces / columns;
		cout << "Puzzle size is " << rows << " x " << columns << " (rows, columns)" << endl;
	}

	// construct left edge
	cout << "Constructing left edge" << endl;
	cursor = completedPuzzle[0][0];  // reset cursor
	for(int i = 1; i < rows; i++) {
		if(cursor->edges[cursor->downIndex()].isEdge) {
			cout << "ERROR: unexpected edge encountered" << endl;
			return;
		}

		PieceMatch matchingPiece = match(cursor, cursor->downIndex(), true, verbose);
		cursor = matchingPiece.piece;
		if(cursor == nullptr) {
			cout << "ERROR: no valid matches found" << endl;
			return;
		}

		cout << "Match: piece " << cursor->number << endl;
		completedPuzzle.push_back(vector<PuzzlePiece*>());
		completedPuzzle[i].push_back(cursor);
		cursor->isConnected = true;
		cursor->rightIndex = PuzzlePiece::nextIndex(matchingPiece.edgeIndex);
		cursor->correctionShiftUp = -matchingPiece.match.shift;  // rotate 180 degrees
		cursor->correctionAngleUp = matchingPiece.match.theta;
	}

	// fill in the rest of the puzzle
	for(int i = 1; i < rows; i++) {
		for(int j = 1; j < columns; j++) {
			PuzzlePiece *leftPiece = completedPuzzle[i][j-1];
			PuzzlePiece *upPiece = completedPuzzle[i-1][j];
			bool noEdges = i < rows - 1 &&  j < columns - 1;
			vector<PieceMatch> matchingPieces = match2(leftPiece, leftPiece->rightIndex, upPiece, upPiece->downIndex(), noEdges, verbose);
			cursor=matchingPieces[0].piece;
			if(matchingPieces[0].piece == nullptr || matchingPieces[1].piece == nullptr) {
				cout << "ERROR: no match found" << endl;
				return;
			}

			cout << "Match: piece " << cursor->number << endl;
			completedPuzzle[i].push_back(cursor);
			cursor->isConnected = true;
			cursor->rightIndex = PuzzlePiece::oppIndex(matchingPieces[0].edgeIndex);
			cursor->correctionShiftLeft = Point(-matchingPieces[0].match.shift.y, matchingPieces[0].match.shift.x); // rotate clockwise 90deg
			cursor->correctionShiftUp = -matchingPieces[1].match.shift;  // rotate 180 degrees
			cursor->correctionAngleLeft = matchingPieces[0].match.theta;
			cursor->correctionAngleUp = matchingPieces[1].match.theta;
		}
	}
	cout << "Puzzle completed!" << endl;
}

// print piece numbers in solved configuration
void Puzzle::print() {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
			cout << setw(4) << completedPuzzle[i][j]->number;
		}
		cout << endl;
	}
}

void Puzzle::display(bool verbose) {

	// create img for final puzzle
	double puzzleWidth = 0;
	double puzzleHeight = 0;
	for(int i = 0; i < completedPuzzle[0].size(); i++) {
		puzzleWidth += completedPuzzle[0][i]->width();
	}
	for(int i = 0; i < completedPuzzle.size(); i++) {
		puzzleHeight += completedPuzzle[i][0]->height();
	}
	Mat completedPuzzleImg = Mat::zeros(puzzleHeight * 1.05, puzzleWidth * 1.05, completedPuzzle[0][0]->img.type());

	vector<Point> redDots;  // for debug
	vector<Point> blueDots;

	// shift / rotate each piece and copy to completed puzzle img
	for(int col = 0; col < columns; col++) {
		for(int row = 0; row < rows; row++) {

			PuzzlePiece *cursor = completedPuzzle[row][col];
			cout << "Displaying piece " << cursor->number << endl;
			PuzzlePiece *leftNeighbor = nullptr;
			PuzzlePiece *upNeighbor = nullptr;

			// rotation corrections
			// (rotate first bc shift depends on location of edge midpoints)
			double theta = 0;
			if(row == 0 && col == 0) {  // top left corner
				theta = (cursor->edges[cursor->leftIndex()].rotCorrection + cursor->edges[cursor->upIndex()].rotCorrection) / 2;
			}  else if(col == 0) {  // left edge
				upNeighbor = completedPuzzle[row-1][col];
				theta = cursor->edges[cursor->leftIndex()].rotCorrection;
				cout << "left edge rot correction: " << theta << endl;
			} else if(row == 0) {  // top edge
				leftNeighbor = completedPuzzle[row][col-1];
				theta = cursor->edges[cursor->upIndex()].rotCorrection;
				cout << "top edge rot correction: " << theta << endl;
				cout << "up index: " << cursor->upIndex() << endl;
				cout << "piece number: " << cursor->number << endl;
			} else {  // most pieces
				upNeighbor = completedPuzzle[row-1][col];
				leftNeighbor = completedPuzzle[row][col-1];
				theta = (cursor->correctionAngleLeft + leftNeighbor->finalCorrectionAngle + cursor->correctionAngleUp + upNeighbor->finalCorrectionAngle) / 2;
			}
			cursor->finalCorrectionAngle = theta;

			// rotate about the center of the piece
			cursor->rotate(cursor->center(), cursor->rotationAngle());  // two steps bc of special behavior when angle = 90, 180, 270
			cursor->rotate(cursor->center(), theta);

			if(verbose) {
				cout << "Correction angle: " << theta << endl;
				cout << "Show after just rotation:" << endl;
				namedWindow("pieces");
				imshow("pieces", cursor->img);
				waitKey(0);
			}

			// calculate translation
			Point shift;
			if(row == 0 && col == 0) {  // top left corner
				shift = -cursor->core.tl() + Point(cursor->edges[cursor->leftIndex()].shiftCorrection, cursor->edges[cursor->upIndex()].shiftCorrection);
			}  else if(col == 0) {  // left edge
				int shiftX = -cursor->edges[3].midpoint.x + cursor->edges[cursor->leftIndex()].shiftCorrection;
				int shiftY = upNeighbor->edges[2].midpoint.y - cursor->edges[0].midpoint.y + cursor->correctionShiftUp.y;
				shift = Point(shiftX, shiftY);
				cout << "left edge shift corrections: " << endl;
				cout << "x edge: " << cursor->edges[cursor->leftIndex()].shiftCorrection << endl;
				cout << "y: " << cursor->correctionShiftUp.y << endl;
				cout << "cursor midpoint x: " << -cursor->edges[3].midpoint.x << endl;
				cout << "up neighbor midpoint: " << upNeighbor->edges[2].midpoint.y << endl;
				cout << "midpoint y: " << cursor->edges[0].midpoint.y << endl;
			} else if(row == 0) {  // top edge
				int shiftX = leftNeighbor->edges[1].midpoint.x - cursor->edges[3].midpoint.x + cursor->correctionShiftLeft.x;
				int shiftY = -cursor->edges[0].midpoint.y + cursor->edges[cursor->upIndex()].shiftCorrection;
				shift = Point(shiftX, shiftY);
				cout << "top edge shift corrections: " << endl;
				cout << "x correction: " << cursor->correctionShiftLeft.x << endl;
				cout << "y correction: " << cursor->edges[cursor->upIndex()].shiftCorrection << endl;
				cout << "cursor midpoint y: " << cursor->edges[0].midpoint.y << endl;
				cout << "left neighbor midpoint: " << leftNeighbor->edges[1].midpoint.x << endl;
				cout << "cursor midpoint: " << cursor->edges[3].midpoint.x << endl;
			} else {  // most pieces
				Point shiftUp = upNeighbor->edges[2].midpoint - cursor->edges[0].midpoint + cursor->correctionShiftUp;
				Point shiftLeft = leftNeighbor->edges[1].midpoint - cursor->edges[3].midpoint + cursor->correctionShiftLeft;
				shift = (shiftUp + shiftLeft) / 2;
			}
			cursor->shift(shift);  // translate the midpoints and core
			cout << "final shift: " << shift << endl; // debug

			if(verbose) {  // debug: draw midpoints and target locations
				blueDots.push_back(cursor->edges[0].midpoint);
				blueDots.push_back(cursor->edges[3].midpoint);
				if(upNeighbor != nullptr) {
					redDots.push_back(upNeighbor->edges[2].midpoint + cursor->correctionShiftUp);
				}
				if(leftNeighbor != nullptr) {
					redDots.push_back(leftNeighbor->edges[1].midpoint + cursor->correctionShiftLeft);
				}
			}

			// create mask shaped like the piece
			Mat mask = Mat::zeros(cursor->img.size(), cursor->img.type());
			vector<vector<Point>> outline_vec = {cursor->outline};
			drawContours(mask, outline_vec, -1, Scalar(255, 255, 255), -1);  // thickness=-1 fills in the contour
			if(verbose) {
				cout << "show mask" << endl;
				imshow("pieces", mask);
				waitKey(0);
				destroyWindow("pieces");
			}

			// copy piece to final puzzle image
			Rect outlineBox = boundingRect(cursor->outline);
			Rect destinationBox = Rect(outlineBox.tl() + shift, outlineBox.size());
			cout << "outline box TL: " << outlineBox.tl() << endl;
			// clip target region so it doesn't go off edge of image
			Rect clippedDestBox = destinationBox & Rect(Point(0, 0), completedPuzzleImg.size());
			Rect clippedOutlineBox = Rect(destinationBox.tl() - shift, clippedDestBox.size());
			cursor->img(clippedOutlineBox).copyTo(completedPuzzleImg(clippedDestBox), mask(clippedOutlineBox));
			if(verbose) {
				cout << "Show puzzle with new piece added:" << endl;
				namedWindow("completed puzzle");
				imshow("completed puzzle", completedPuzzleImg);
				waitKey(0);
			}
		}
	}

	for(Point p: redDots) {  // debug
		circle(completedPuzzleImg, p, 15, Scalar(0, 0, 255), -1);
	}
	for(Point p: blueDots) {
		circle(completedPuzzleImg, p, 15, Scalar(255, 0, 0), -1);
	}

	// show completed puzzle
	imshow("completed puzzle", completedPuzzleImg);
	waitKey(0);

	// rotate the entire puzzle based on user input
	string fullPuzzleRotStr;
	cout << "Enter degrees of counterclockwise rotation: 0, 90, 180, or 270" << endl;
	cin >> fullPuzzleRotStr;
	int fullPuzzleRot = stoi(fullPuzzleRotStr);
	Mat t_puzz;
	if(fullPuzzleRot == 90) {
		t_puzz = getRotationMatrix2D(Point(max(puzzleWidth, puzzleHeight)/2, max(puzzleWidth, puzzleHeight)/2), fullPuzzleRot, 1);
		warpAffine(completedPuzzleImg, completedPuzzleImg, t_puzz, Size(puzzleHeight, puzzleWidth));
	} else if(fullPuzzleRot == 180) {
		t_puzz = getRotationMatrix2D(Point(puzzleWidth/2, puzzleHeight/2), fullPuzzleRot, 1);
		warpAffine(completedPuzzleImg, completedPuzzleImg, t_puzz, Size(puzzleWidth, puzzleHeight));
	} else if(fullPuzzleRot == 270) {
		t_puzz = getRotationMatrix2D(Point(min(puzzleWidth, puzzleHeight)/2, min(puzzleWidth, puzzleHeight)/2), fullPuzzleRot, 1);
		warpAffine(completedPuzzleImg, completedPuzzleImg, t_puzz, Size(puzzleHeight, puzzleWidth));
	}
	imshow("completed puzzle", completedPuzzleImg);
	waitKey(0);

	// show piece numbers
	for(int col = 0; col < columns; col++) {
			for(int row = 0; row < rows; row++) {
				PuzzlePiece *cursor = completedPuzzle[row][col];
				Point textLoc = (cursor->core.tl() + cursor->core.br()) / 2;
				if(fullPuzzleRot != 0) {
					textLoc = rotatePoint(textLoc, t_puzz);
				}
				putText(completedPuzzleImg, to_string(cursor->number), textLoc, FONT_HERSHEY_SIMPLEX, 5, Scalar(0, 0, 0), 20);
				putText(completedPuzzleImg, to_string(cursor->number), textLoc, FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 255, 255), 5);
			}
	}
	imshow("completed puzzle", completedPuzzleImg);
	waitKey(0);
	destroyWindow("completed puzzle");
}

Point rotatePoint(Point p, Mat t) {
	double x = t.at<double>(0, 0) * p.x + t.at<double>(0, 1) * p.y + t.at<double>(0, 2);
	double y = t.at<double>(1, 0) * p.x + t.at<double>(1, 1) * p.y + t.at<double>(1, 2);
	return Point(x, y);
}

// test all edges against all others
// TODO: distance from next best score. set the actual match and call isConnected() to get next match
// TODO: if fail, show best match
void Test::testAllEdgePairs(Puzzle myPuzzle, bool secondBest) {

	// pieces here are by number, NOT index
	vector<vector<vector<int>>> expectedMatches;
	expectedMatches.push_back({{2, 2}, {}, {4, 2}, {14, 3}});
	expectedMatches.push_back({{9, 0}, {}, {1, 0}, {12, 0}});
	expectedMatches.push_back({{10, 3}, {16, 2}, {11, 2}, {}});
	expectedMatches.push_back({{}, {5, 2}, {1, 2}, {}});
	expectedMatches.push_back({{7, 2}, {14, 0}, {4, 1}, {}});
	expectedMatches.push_back({{14, 1}, {7, 1}, {15, 3}, {16, 0}});
	expectedMatches.push_back({{8, 2}, {6, 1}, {5, 0}, {}});
	expectedMatches.push_back({{}, {15, 0}, {7, 0}, {}});
	expectedMatches.push_back({{2, 0}, {10, 1}, {}, {}});
	expectedMatches.push_back({{}, {9, 1}, {12, 3}, {3, 0}});
	expectedMatches.push_back({{}, {}, {3, 2}, {13, 0}});
	expectedMatches.push_back({{2, 3}, {14, 2}, {16, 3}, {10, 2}});
	expectedMatches.push_back({{11, 3}, {16, 1}, {15, 2}, {}});
	expectedMatches.push_back({{5, 1}, {6, 0}, {12, 1}, {1, 3}});
	expectedMatches.push_back({{8, 1}, {}, {13, 2}, {6, 2}});
	expectedMatches.push_back({{6, 3}, {13, 1}, {3, 1}, {12, 2}});

	ofstream file("match_scores.txt");
	vector<double> scoreRatios;

	int numPass = 0;
	int numFail = 0;
	for(int n = 0; n < myPuzzle.numPieces; n++) {
		for(int i = 0; i < 4; i++) {  // TEST
			if(expectedMatches[n][i].size() == 0) continue;
			PuzzlePiece *cursor = &myPuzzle.pieces[n];
			PieceMatch matchingPiece = myPuzzle.match(cursor, i, false, false);
			PuzzlePiece *piece = matchingPiece.piece;

			// piece->isConnected = false;  // unnecessary, isConnected is set in assemble()

			if(piece->number == expectedMatches[n][i][0] && matchingPiece.edgeIndex == expectedMatches[n][i][1]) {
				file << "PASS: ";
				numPass++;
			} else {
				file << "FAIL: ";
				numFail++;
			}
			file << "Piece " << cursor->number << ", index " << i << " matches piece " << piece->number << ", index " << matchingPiece.edgeIndex << "(score " << matchingPiece.match.score << ")" << endl;

			if(secondBest) {
				myPuzzle.pieces[expectedMatches[n][i][0]-1].isConnected = true;
				double firstScore = matchingPiece.match.score;
				matchingPiece = myPuzzle.match(cursor, i, false, false);
				piece = matchingPiece.piece;
				myPuzzle.pieces[expectedMatches[n][i][0]-1].isConnected = false;  // disconnect
				double scoreRatio = matchingPiece.match.score / firstScore;
				scoreRatios.push_back(scoreRatio);
				file << "second best match: piece " << piece->number << ", index " << matchingPiece.edgeIndex << "(score " << matchingPiece.match.score << ", ratio " << scoreRatio << ")" << endl;
			}
		}
	}

	if(secondBest) {
		sort(scoreRatios.begin(), scoreRatios.end());
		for(double s: scoreRatios) {
			cout << s << " ";
		}
		cout << endl;
	}

	file << numPass << "/" << numPass+numFail << " PASS, " << numFail << "/" << numPass+numFail << " FAIL" << endl;
	file.close();
}

void Test::displayEdgeMatches(Puzzle myPuzzle) {
	PuzzlePiece *pieces = myPuzzle.pieces;

	// values are: pieceIdx, edgeIdx, pieceIdx, edgeIdx
	// uses piece numbers, NOT indices
	vector<vector<int>> idxs;

	// close to false positive
	idxs.push_back({12, 1, 14, 2});
	idxs.push_back({12, 1, 10, 2});

	// cout << "p to go back" << endl;
	for (int i = 0; i < idxs.size(); ) {
		vector<int> v = idxs[i];
		EdgeOfPiece::matchEdges(pieces[v[0]-1].edges[v[1]], pieces[v[2]-1].edges[v[3]], true);

//		string nextStr;
//		cin >> nextStr;
//		if(nextStr == "p") i--;
//		else i++;
		i++;
	}
}

void Test::prototyping() {

	/*
	// outline: bilateral filter
	Mat smallImg;
	resize(img, smallImg, Size(img.size[1]/5, img.size[0]/5));
	Mat img_hsv;
	cvtColor(smallImg, img_hsv, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(img_hsv, channels);
	Mat channel = channels[1];
	Mat filteredImg;
	bilateralFilter(channel, filteredImg, 10, 80, 16, BORDER_DEFAULT); // I think larger neighborhoods take longer
	Mat color_mask;
	Canny(filteredImg, color_mask, 50, 100);
	resize(color_mask, color_mask, img.size());
	if(verbose) {
		namedWindow("bilateral");
		imshow("bilateral", smallImg);
		waitKey(0);
		imshow("bilateral", channel);
		waitKey(0);
		imshow("bilateral", filteredImg);
		waitKey(0);
		imshow("bilateral", color_mask);
		waitKey(0);
		destroyWindow("bilateral");
	}
	*/

	/*
	// outline: adaptive threshold
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(img_hsv, channels);
	Mat channel = channels[1]; // saturation
	channel = 255 - channel; // invert
	Mat smallImg;
	resize(channel, smallImg, Size(channel.size[1]/5, channel.size[0]/5));
	Mat filteredImg;
	bilateralFilter(smallImg, filteredImg, 10, 80, 10, BORDER_DEFAULT); // I think larger neighborhoods take longer
	resize(filteredImg, filteredImg, channel.size());
	Mat dilated_piece;
	adaptiveThreshold(filteredImg, dilated_piece, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 301, -10);  // blockSize must be odd
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(30, 30));  // err on the side of more
	dilate(dilated_piece, dilated_piece, kernel);
	Mat threshold;
	adaptiveThreshold(filteredImg, threshold, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 301, 0);  // blockSize must be odd
	Mat color_mask;
	bitwise_and(dilated_piece, threshold, color_mask);
	if(verbose) {
		namedWindow("adaptive");
		imshow("adaptive", img);
		waitKey(0);
		imshow("adaptive", channel);
		waitKey(0);
		imshow("adaptive", filteredImg);
		waitKey(0);
		imshow("adaptive", dilated_piece);
		waitKey(0);
		imshow("adaptive", threshold);
		waitKey(0);
		imshow("adaptive", color_mask);
		waitKey(0);
		destroyWindow("adaptive");
	}
	*/

	/*
	// outline: otsu
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(img_hsv, channels);
	Mat channel = channels[1]; // saturation
	channel = 255 - channel; // invert bc piece has lower saturation
	Mat smallChannel;
	resize(channel, smallChannel, Size(channel.size[1]/5, channel.size[0]/5));
	medianBlur(smallChannel, smallChannel, 255);
	Mat background;
	resize(smallChannel, background, channel.size());
	Mat flat = channel - background;
	Mat color_mask; // apply otsu
	threshold(flat, color_mask, 0, 255, THRESH_BINARY+THRESH_OTSU);  //why red underline
	if(verbose) {
		namedWindow("otsu");
		imshow("otsu", img);
		waitKey(0);
		imshow("otsu", channel);
		waitKey(0);
		imshow("otsu", background);
		waitKey(0);
		imshow("otsu", flat);
		waitKey(0);
		imshow("otsu", color_mask);
		waitKey(0);
		// imshow("otsu", color_mask);
		// waitKey(0);
		destroyWindow("otsu");
	}
	*/
}
