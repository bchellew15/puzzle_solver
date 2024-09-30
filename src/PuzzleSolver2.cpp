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

	bool process_verbose = false;
	bool match_verbose = false;
	bool display_verbose = false;

	int numPieces = 16;
	cout << numPieces << " pieces" << endl;

	// load images
	Mat images[numPieces];
	PuzzlePiece pieces[numPieces];
	string dir = "/Users/blakechellew/Documents/Code/workspace/PuzzleSolver2/Pieces_16_darkgreen/Piece";
	for( int i = 0; i < numPieces; i++) {
		string filename = dir + to_string(i+1) + ".jpeg";
		images[i] = imread(filename);
		pieces[i] = PuzzlePiece(images[i], i);
	}

	// create a Puzzle
	Puzzle myPuzzle = Puzzle(numPieces, pieces);

	chrono::time_point<chrono::steady_clock> start_time = chrono::steady_clock::now();
	myPuzzle.process(process_verbose);
	chrono::time_point<chrono::steady_clock> end_time = chrono::steady_clock::now();
	cout << "Processing time: " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << endl;

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

int EdgeOfPiece::edgeImgBuffer = 10;
double EdgeOfPiece::edgeShrinkFactor = 5;
int EdgeOfPiece::pixelShift = 5 / EdgeOfPiece::edgeShrinkFactor;
double PuzzlePiece::scalingLength = 0;
double PuzzlePiece::avgBrightness = 0;


// fit line to decide if the edge is flat.
// if yes, calculate rotation and vertical shift required to line up the edge with border of puzzle.
// if no, create raster images of the edge.
void EdgeOfPiece::processEdge() {
	vector<double> fittedLine;
	fitLine(edge, fittedLine, DIST_L2, 0.0, .01, .01);

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
		thetaCorrection = atan(vy / vx) * 180 / 3.14;
		shiftCorrection = -(y0 - vy/vx*x0);
		isEdge = true;
		return;
	}
	isEdge = false;

	// create raster images of the edge
	Rect edgeBound = boundingRect(edge);
	edgeImg = Mat::zeros((edgeBound.height + 2*EdgeOfPiece::edgeImgBuffer) / EdgeOfPiece::edgeShrinkFactor, (edgeBound.width + 2*EdgeOfPiece::edgeImgBuffer) / EdgeOfPiece::edgeShrinkFactor, CV_8UC1);
	rasterShift = (- Point(edgeBound.x, edgeBound.y) + Point(EdgeOfPiece::edgeImgBuffer, EdgeOfPiece::edgeImgBuffer)) / EdgeOfPiece::edgeShrinkFactor;
	for(Point p: edge) {
		Point circleLoc = p / EdgeOfPiece::edgeShrinkFactor + rasterShift;
		circle(edgeImg, circleLoc, EdgeOfPiece::edgeImgBuffer / EdgeOfPiece::edgeShrinkFactor, 255, -1);
	}
	// raster images of rotated edges
	for(int theta = -4; theta <= 4; theta +=2) {
		rotEdgeImgDegrees.push_back(theta);
	}
	Point rotationCenter = Point(edgeImg.cols/2, edgeImg.rows/2);
	for(double deg: rotEdgeImgDegrees) {
		Mat rotatedEdgeImg;
		Mat rot_t = getRotationMatrix2D(rotationCenter, deg+180, 1);
		warpAffine(edgeImg, rotatedEdgeImg, rot_t, edgeImg.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
		rotEdgeImgs.push_back(rotatedEdgeImg);
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
	Mat img_hsv;
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

	// create color mask
	Mat color_mask;
	double hueBuffer = 2;  // fraction denominator of range that is added to each end
	double satBuffer = 2;
	double valueBuffer = 2;
	Scalar colorLowerBound = Scalar(max(0.0, h_channel_min - h_channel_width/hueBuffer), max(0.0, s_channel_min - s_channel_width/satBuffer), max(0.0, v_channel_min - v_channel_width/valueBuffer));
	Scalar colorUpperBound = Scalar(min(255.0, h_channel_max + h_channel_width/hueBuffer), min(255.0, s_channel_max + s_channel_width/satBuffer), min(255.0, v_channel_max + v_channel_width/valueBuffer));
	inRange(img_hsv, colorLowerBound, colorUpperBound, color_mask);
	color_mask = 255 - color_mask;  // invert
	if(verbose) {
		namedWindow("mask");
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

	// adjust the brightness level
	if(PuzzlePiece::avgBrightness == 0) {
		PuzzlePiece::avgBrightness = v_avg;
	}
	img = img * PuzzlePiece::avgBrightness / v_avg;

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
	midpoints[0] = core.tl() + Point(core.width/2, 0);
	midpoints[1] = core.tl() + Point(core.width, core.height/2);
	midpoints[2] = core.tl() + Point(core.width/2, core.height);
	midpoints[3] = core.tl() + Point(0, core.height/2);
	for(int i = 0; i < 4; i++) {
		for(Point &p: edges[i].edge) {
			p -= midpoints[i];
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
				imshow("grey", edges[0].edgeImg);
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
	return (index - 1) % 4;
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

// lower score is better.
// "shift" and "theta" are corrections for edge 2.
EdgeMatch EdgeOfPiece::matchEdges(EdgeOfPiece edge1, EdgeOfPiece edge2, bool verbose) {

	int minHeight = min(edge1.edgeImg.rows, edge2.edgeImg.rows);
	int maxHeight = max(edge1.edgeImg.rows, edge2.edgeImg.rows);
	int h_intervals = (maxHeight - minHeight) / pixelShift;
	int minWidth = min(edge1.edgeImg.cols, edge2.edgeImg.cols);
	int windowWidth = minWidth * 9 / 10;
	int maxWidth = max(edge1.edgeImg.cols, edge2.edgeImg.cols);
	int w_intervals = (maxWidth - windowWidth) / pixelShift;

	double minScore;
	int bestTheta;
	Point bestShift;
	bool firstScore = true;
	Mat best_e1; // for display only
	Mat best_e2; // for display only

	for(int i = 0; i < edge2.rotEdgeImgDegrees.size(); i++) {

		double theta = edge2.rotEdgeImgDegrees[i];
		Mat rotEdgeImg = edge2.rotEdgeImgs[i];

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

				if(edge1.edgeImg.rows == minHeight) {
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

				if(edge1.edgeImg.cols == minWidth) {
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
				Mat e1 = edge1.edgeImg.rowRange(e1_row_min, e1_row_max).colRange(e1_col_min, e1_col_max);
				Mat e2 = rotEdgeImg.rowRange(e2_row_min, e2_row_max).colRange(e2_col_min, e2_col_max);

				double score = edgeComparisonScore(e1, e2);
				if(firstScore) {
					firstScore = false;
					minScore = score;
					bestTheta = theta;
					// first flip the raster shift across center row bc working with flipped img
					bestShift = Point(edge2.rasterShift.x, rotEdgeImg.rows - edge2.rasterShift.y) + Point(e1_col_min - e2_col_min, e1_row_min - e2_row_min) - edge1.rasterShift;
					best_e1 = e1;  // for display
					best_e2 = e2;
					// cout << "first final correction: " << Point(e1_col_min - e2_col_min, e1_row_min - e2_row_min) << endl;
				} else if(score < minScore) {
					minScore = score;
					bestTheta = theta;
					bestShift = Point(edge2.rasterShift.x, rotEdgeImg.rows - edge2.rasterShift.y) + Point(e1_col_min - e2_col_min, e1_row_min - e2_row_min) - edge1.rasterShift;
					best_e1 = e1;  // for display
					best_e2 = e2;
					// cout << "new final correction: " << Point(e1_col_min - e2_col_min, e1_row_min - e2_row_min) << endl;
				}
			}
		}
	}

	if(verbose) {
		cout << "best theta: " << bestTheta << endl;

		Mat bothEdges = Mat::zeros(best_e1.size(), CV_8UC3);
		Mat channels[3] = {best_e1, Mat::zeros(best_e1.size(), CV_8UC1), best_e2};
		merge(channels, 3, bothEdges);

		namedWindow("edgeMatch");
		imshow("edgeMatch", bothEdges);
		waitKey(0);
		destroyWindow("edgeMatch");
	}

	// cout << "best shift: " << bestShift << endl;
	// cout << "undo flip: " << Point(0, edge2.edgeImg.rows - 2 * EdgeOfPiece::edgeImgBuffer) << endl;
	// cout << "piece 2 raster shift: " << - edge2.rasterShift << endl;
	// cout << "piece 1 raster shift: " << edge1.rasterShift << endl;

	EdgeMatch bestMatch;
	bestMatch.score = minScore;
	bestMatch.theta = bestTheta;
	// cout << "theta of best score: " << bestTheta << endl;
	bestMatch.shift = bestShift * edgeShrinkFactor;
	return bestMatch;

	// ISSUE: these print blanks
	// cout << e1.at<uchar>(0, 0) << endl;
	// cout << e2.at<uchar>(e2.rows-1, e2.cols-1) << endl;

	// prep by rotating one edge (by flipping twice)
	vector<Point> flippedEdge = vector<Point>(edge2.edge.size());
	for(int i = 0; i < edge2.edge.size(); i++) {
		flippedEdge[i].x = -edge2.edge[i].x;
		flippedEdge[i].y = -edge2.edge[i].y;
	}

	// show the edges being compared
	Mat blank = Mat::zeros(3000, 3000, CV_8UC3);
	vector<vector<Point>> twoEdges = {edge1.edge, flippedEdge};
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

	if(piece->edges[edgeIndex].isEdge) {
		cout << "ERROR: calling match() on an edge piece" << endl;
		bestMatch.piece = nullptr;
		return bestMatch;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdge) continue; // skip if it's an edge
			EdgeMatch currentMatch = EdgeOfPiece::matchEdges(piece->edges[edgeIndex], pieces[i].edges[j], verbose);

			cout << "Piece " << piece->number << " scores " << currentMatch.score << " against index " << j << " of piece " << i+1 << endl;

			if(firstMatch) {
				bestMatchScore = currentMatch.score;
				bestMatch.match = currentMatch;
				bestMatch.piece = &pieces[i];
				bestMatch.edgeIndex = j;
				firstMatch = false;
			}
			else if(currentMatch.score < bestMatchScore) { // low score is best
				bestMatchScore = currentMatch.score;
				bestMatch.match = currentMatch;
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

vector<PieceMatch> Puzzle::match2(PuzzlePiece *leftPiece, int edgeIndexOfLeft, PuzzlePiece *upPiece, int edgeIndexOfUp, bool verbose) {

	PieceMatch bestMatchLeft;
	PieceMatch bestMatchUp;
	bool firstMatch = true;
	double bestMatchScore;

	if(leftPiece->edges[edgeIndexOfLeft].isEdge || upPiece->edges[edgeIndexOfUp].isEdge) {
		cout << "ERROR: calling match() on an edge piece" << endl;
		bestMatchLeft.piece = nullptr;
		vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
		return matches;
	}

	for(int i = 0; i < numPieces; i++) {
		if(pieces[i].isConnected) continue; // skip if already connected
		for(int j = 0; j < 4; j++) {
			if(pieces[i].edges[j].isEdge || pieces[i].edges[(j+1)%4].isEdge) continue; // skip if either connection is an edge
			PieceMatch matchLeft;
			matchLeft.match = EdgeOfPiece::matchEdges(leftPiece->edges[edgeIndexOfLeft], pieces[i].edges[j], verbose);
			PieceMatch matchUp;
			matchUp.match = EdgeOfPiece::matchEdges(upPiece->edges[edgeIndexOfUp], pieces[i].edges[(j+1)%4], verbose);
			double score = matchLeft.match.score + matchUp.match.score;
			cout << "Pieces " << leftPiece->number << " and " << upPiece->number <<
					" score " << score << " against Piece " << pieces[i].number <<
					" with right index " << (j+2)%4  << endl;

			if(firstMatch) {
				firstMatch = false;
				bestMatchScore = score;
				bestMatchLeft = matchLeft;
				bestMatchUp = matchUp;
				bestMatchLeft.piece = &pieces[i];
				bestMatchLeft.edgeIndex = j;
			}
			else if(score < bestMatchScore) { // low score is best
				bestMatchScore = score;
				bestMatchLeft = matchLeft;
				bestMatchUp = matchUp;
				bestMatchLeft.piece = &pieces[i];
				bestMatchLeft.edgeIndex = j;
			}
		}
	}

	if(firstMatch) {
		cout << "ERROR: remaining pieces are edges only." << endl;
		bestMatchLeft.piece = nullptr;
		vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
		return matches;
	}

	cout << "Pieces " << leftPiece->number << " and " << upPiece->number << " match Piece "
			<< bestMatchLeft.piece->number << " with right index " << bestMatchLeft.edgeIndex  << endl;

	vector<PieceMatch> matches = {bestMatchLeft, bestMatchUp};
	return matches;
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
	return (core.tl() + core.br()) / 2;
}

// note: outline should be oriented counter clockwise bc outer contour. but, I don't think that guarantees sequential order.
vector<Point> PuzzlePiece::constructEdge(int firstIdx, int secondIdx) {
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
	for(Point &p: outline) { p *= factor; }
	core = Rect(core.x * factor, core.y * factor, core.width * factor, core.height * factor);
	for(Point &p: midpoints) { p *= factor; }
}

// image, outline, core, edges
void PuzzlePiece::shift(Point s, Size newSize) {
	double t_values[] = {1, 0, (double)s.x,
										0, 1, (double)s.y};
	Mat t_shift = Mat(2, 3, DataType<double>::type, t_values); // not sure about that data type
	warpAffine(img, img, t_shift, newSize);  // may want to preserve the original image...
	for(Point &p: outline) { p += s; }
	core = Rect(core.x + s.x, core.y + s.y, core.width, core.height);
	for(Point &p: midpoints) { p += s; }
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

	// rotate core and midpoints, assuming rotationCenter is center of piece
	// note: if theta = 180, core and midpoints stay the same
	if(theta == 90 || theta == 270) {
		core = Rect(core.tl().x + core.width/2 - core.height/2, core.tl().y + core.height/2 - core.width/2, core.height, core.width);
		midpoints[0] = core.tl() + Point(core.width/2, 0);
		midpoints[1] = core.tl() + Point(core.width, core.height/2);
		midpoints[2] = core.tl() + Point(core.width/2, core.height);
		midpoints[3] = core.tl() + Point(0, core.height/2);
	} else if(theta < 90) {
		for(Point &p: midpoints) {
			double temp_x = p.x;
			p.x = t.at<double>(0, 0) * p.x + t.at<double>(0, 1) * p.y + t.at<double>(0, 2);
			p.y = t.at<double>(1, 0) * temp_x + t.at<double>(1, 1) * p.y + t.at<double>(1, 2);
		}

		// rotate the shift corrections
		Mat t_zero = getRotationMatrix2D(Point(0, 0), theta, 1);
		Point sl = correctionShiftLeft;
		correctionShiftLeft.x = t_zero.at<double>(0, 0) * sl.x + t_zero.at<double>(0, 1) * sl.y + t_zero.at<double>(0, 2);
		correctionShiftLeft.y = t_zero.at<double>(1, 0) * sl.x + t_zero.at<double>(1, 1) * sl.y + t_zero.at<double>(1, 2);
		Point su = correctionShiftUp;
		correctionShiftUp.x = t_zero.at<double>(0, 0) * su.x + t_zero.at<double>(0, 1) * su.y + t_zero.at<double>(0, 2);
		correctionShiftUp.y = t_zero.at<double>(1, 0) * su.x + t_zero.at<double>(1, 1) * su.y + t_zero.at<double>(1, 2);

		// rotate edge corrections
		for(EdgeOfPiece &e: edges) {
			if(e.isEdge) {
				e.shiftCorrection = e.shiftCorrection * cos(theta);
			}
		}
	}
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
	completedPuzzle[0][0]->rightIndex = completedPuzzle[0][0]->rootRightIndex();  // set orientation of top left corner
	cout << "Root is piece " << completedPuzzle[0][0]->number << endl;

	cout << "Start matching." << endl;

	// construct top edge
	cout << "Constructing top edge." << endl;
	PuzzlePiece *cursor = completedPuzzle[0][0];
	while(!cursor->edges[cursor->rightIndex].isEdge && completedPuzzle[0].size() < numPieces) {
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
		cursor->correctionShiftLeft = Point(-matchingPiece.match.shift.y, matchingPiece.match.shift.x);
		cursor->correctionThetaLeft = matchingPiece.match.theta;
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
		if(cursor->edges[cursor->downIndex()].isEdge) {
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
		cursor->correctionShiftUp = -matchingPiece.match.shift; // 180 degree rotation
		cursor->correctionThetaUp = matchingPiece.match.theta;
	}

	// fill in the rest:
	for(int i = 1; i < rows; i++) {
		for(int j = 1; j < columns; j++) {
			PuzzlePiece *leftPiece = completedPuzzle[i][j-1];
			PuzzlePiece *upPiece = completedPuzzle[i-1][j];
			vector<PieceMatch> matchingPieces = match2(leftPiece, leftPiece->rightIndex, upPiece, upPiece->downIndex(), verbose);
			if(matchingPieces[0].piece == nullptr || matchingPieces[1].piece == nullptr) {
				cout << "ERROR: no match found" << endl;
				return;
			}
			cursor=matchingPieces[0].piece;
			completedPuzzle[i].push_back(cursor);
			cursor->isConnected = true;
			cursor->rightIndex = PuzzlePiece::oppIndex(matchingPieces[0].edgeIndex);
			cursor->correctionShiftLeft = Point(-matchingPieces[0].match.shift.y, matchingPieces[0].match.shift.x);  // // rotate clockwise 90deg
			cursor->correctionShiftUp = -matchingPieces[1].match.shift; // 180 degree rotation
			cursor->correctionThetaLeft = matchingPieces[0].match.theta;
			cursor->correctionThetaUp = matchingPieces[1].match.theta;
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
	double puzzleWidth = 0;
	for(int i = 0; i < completedPuzzle[0].size(); i++) {
		puzzleWidth += completedPuzzle[0][i]->width();
	}
	double puzzleHeight = 0;
	for(int i = 0; i < completedPuzzle.size(); i++) {
		puzzleHeight += completedPuzzle[i][0]->height();
	}
	double edgeLength = max(puzzleWidth, puzzleHeight) * 1.05; // leave room for rotating at the end
	Mat completedPuzzleImg = Mat::zeros(edgeLength, edgeLength, completedPuzzle[0][0]->img.type());
	cout << "completed puzzle size: " << completedPuzzleImg.size() << endl;

	// debug
	vector<Point> redDots;
	vector<Point> blueDots;

	// loop through the pieces and copy to completed image
	for(int col = 0; col < columns; col++) {
		for(int row = 0; row < rows; row++) {

 			// todo: the rotation moves the top left corner if pieces are not square. need to account for that.
			// use the new top-left corner in the calculations instead.

			// if attaching left: rotate the left edge centroid. then line it up with the (rotated) centroid of the piece to the left.

			PuzzlePiece *cursor = completedPuzzle[row][col];
			PuzzlePiece *leftNeighbor = nullptr;
			PuzzlePiece *upNeighbor = nullptr;

			cout << "Piece " << cursor->number << endl;

			// rotation corrections
			double correctionTheta = 0;

			// issue: I'm assuming all the top edge and left edge have edges in the right place. if not, accessing those vars will cause errors.
			if(row == 0 && col == 0) {  // top left corner
				correctionTheta = (cursor->edges[cursor->leftIndex()].thetaCorrection + cursor->edges[cursor->upIndex()].thetaCorrection) / 2;
				// todo: correct rotation of the whole puzzle at the end
			}  else if(col == 0) {  // left edge
				upNeighbor = completedPuzzle[row-1][col];
				correctionTheta = cursor->edges[cursor->leftIndex()].thetaCorrection;
			} else if(row == 0) {  // top edge
				leftNeighbor = completedPuzzle[row][col-1];
				correctionTheta = cursor->edges[cursor->upIndex()].thetaCorrection;
			} else {  // most pieces
				upNeighbor = completedPuzzle[row-1][col];
				leftNeighbor = completedPuzzle[row][col-1];
				correctionTheta = (cursor->correctionThetaLeft + leftNeighbor->finalRotationCorrection + cursor->correctionThetaUp + upNeighbor->finalRotationCorrection) / 2;
			}

			// rotate about the center of the piece
			// rotate before scaling so the width / height are correct
			Point rotationCenter = Point(cursor->core.tl().x + cursor->core.width/2, cursor->core.tl().y + cursor->core.height/2);
			cursor->rotate(rotationCenter, cursor->rotationAngle());  // only 90 and 270 cause core to rotate
			cursor->rotate(rotationCenter, correctionTheta);
			cursor->finalRotationCorrection = correctionTheta;
			if(verbose) {
				cout << "correction angle: " << correctionTheta << endl;
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

			// should do in one step; I'm counting on no clipping after the rotation
			// note: could do this check implicitly by waiting until this while loop exits to the other one

			// calculate translation
			// note: rotate before calculating the shift bc depends on the location of core midpoints
			// upNeighbor and leftNeighbor are already calculated during rotation
			Point2f shift; // about converting int to double if I just do Point

			if(row == 0 && col == 0) {  // top left corner
				shift = -cursor->core.tl() + Point(cursor->edges[cursor->leftIndex()].shiftCorrection, cursor->edges[cursor->upIndex()].shiftCorrection);
			}  else if(col == 0) {  // left edge
				int shiftX = - cursor->midpoints[3].x + cursor->edges[cursor->leftIndex()].shiftCorrection;
				int shiftY = upNeighbor->midpoints[2].y - cursor->midpoints[0].y + cursor->correctionShiftUp.y;
				shift = Point(shiftX, shiftY);
			} else if(row == 0) {  // top edge
				int shiftX = leftNeighbor->midpoints[1].x - cursor->midpoints[3].x + cursor->correctionShiftLeft.x;
				int shiftY = - cursor->midpoints[0].y + cursor->edges[cursor->upIndex()].shiftCorrection;
				// cout << "debug: y shift correction: " << cursor->edges[cursor->upIndex()].shiftCorrection << endl;
				shift = Point(shiftX, shiftY);
			} else {  // most pieces
				Point shiftUp = upNeighbor->midpoints[2] - cursor->midpoints[0] + cursor->correctionShiftUp;
				Point shiftLeft = leftNeighbor->midpoints[1] - cursor->midpoints[3] + cursor->correctionShiftLeft;
				shift = (shiftUp + shiftLeft) / 2;
			}

//			if(upNeighbor != nullptr) {
//				cout << "up nieghbor midpoint: " << upNeighbor->midpoints[2] << endl;
//			}
//			if(leftNeighbor != nullptr) {
//				cout << "left neighbor midpoint: " << leftNeighbor->midpoints[1] << endl;
//			}
//			cout << "correction up: " << cursor->correctionShiftUp << endl;
//			cout << "correction left: " << cursor->correctionShiftLeft << endl;

			// translate
			cursor->shift(shift, completedPuzzleImg.size());  // this makes cursor->img permanently much bigger, taking up a lot of RAM...
			if(verbose) {
				cout << "shift: " << shift << endl;
				cout << "show completed transformation" << endl;
				imshow("temp", cursor->img);
				waitKey(0);
			}

			// debug: draw midpoints and goal locations
			if(verbose) {
				blueDots.push_back(cursor->midpoints[0]);
				blueDots.push_back(cursor->midpoints[3]);
				if(upNeighbor != nullptr) {
					redDots.push_back(upNeighbor->midpoints[2] + cursor->correctionShiftUp);
				}
				if(leftNeighbor != nullptr) {
					redDots.push_back(leftNeighbor->midpoints[1] + cursor->correctionShiftLeft);
				}
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

	// debug
	for(Point p: redDots) {
		circle(completedPuzzleImg, p, 15, Scalar(0, 0, 255), -1);
	}
	for(Point p: blueDots) {
		circle(completedPuzzleImg, p, 15, Scalar(255, 0, 0), -1);
	}

	if(checkRotation) cout << "Enter degrees of clockwise rotation" << endl;

	// show completed puzzle
	imshow("temp", completedPuzzleImg);
	waitKey(0);

	if(checkRotation) {
		string rotationStr;
		cin >> rotationStr; // not sure how this interacts with waitKey()
		int fullPuzzleRotation = stoi(rotationStr); // error handling
		Point puzzleCenter = Point(min(puzzleWidth, puzzleHeight)/2, min(puzzleWidth, puzzleHeight)/2);
		Mat t3 = getRotationMatrix2D(puzzleCenter, fullPuzzleRotation, 1);
		// maybe: create new rotated image w reversed dimensions instead of starting with big square image
		warpAffine(completedPuzzleImg, completedPuzzleImg, t3, completedPuzzleImg.size());
		imshow("temp", completedPuzzleImg);
		waitKey(0);
	}

	// show piece numbers
	for(int col = 0; col < columns; col++) {
			for(int row = 0; row < rows; row++) {
				PuzzlePiece *cursor = completedPuzzle[row][col];
				// font size is arbitrary param
				putText(completedPuzzleImg, to_string(cursor->number), (cursor->core.tl() + cursor->core.br())/2, FONT_HERSHEY_SIMPLEX, 5, Scalar(0, 0, 0), 20);
				putText(completedPuzzleImg, to_string(cursor->number), (cursor->core.tl() + cursor->core.br())/2, FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 255, 255), 5);
			}
	}
	imshow("temp", completedPuzzleImg);
	waitKey(0);

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
}
