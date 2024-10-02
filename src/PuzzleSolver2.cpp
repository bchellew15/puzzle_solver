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
	bool display_verbose = true;

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
	Rect edgeBound = boundingRect(edge);
	edgeImg = Mat::zeros((edgeBound.height + 2*EdgeOfPiece::edgeImgBuffer) / EdgeOfPiece::edgeShrinkFactor, (edgeBound.width + 2*EdgeOfPiece::edgeImgBuffer) / EdgeOfPiece::edgeShrinkFactor, CV_8UC1);
	rasterShift = (- Point(edgeBound.x, edgeBound.y) + Point(EdgeOfPiece::edgeImgBuffer, EdgeOfPiece::edgeImgBuffer)) / EdgeOfPiece::edgeShrinkFactor;
	for(Point p: edge) {
		Point circleLoc = p / EdgeOfPiece::edgeShrinkFactor + rasterShift;
		circle(edgeImg, circleLoc, EdgeOfPiece::edgeImgBuffer / EdgeOfPiece::edgeShrinkFactor, 255, -1);
	}
	// raster images of rotated edges
	for(int theta = -4; theta <= 4; theta +=2) {
		rotEdgeImgAngles.push_back(theta);
	}
	Point imgCenter = Point(edgeImg.cols/2, edgeImg.rows/2);
	Point rotationCenter = midpoint / EdgeOfPiece::edgeShrinkFactor + rasterShift;
	for(double deg: rotEdgeImgAngles) {
		Mat rotatedEdgeImg;
		Mat rot1 = getRotationMatrix2D(imgCenter, 180, 1);
		warpAffine(edgeImg, rotatedEdgeImg, rot1, edgeImg.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
		Mat rot2 = getRotationMatrix2D(rotationCenter, deg, 1);
		warpAffine(rotatedEdgeImg, rotatedEdgeImg, rot2, edgeImg.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
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
	Mat xor_mat;
	Mat and_mat;
	bitwise_xor(edge1, edge2, xor_mat);
	bitwise_and(edge1, edge2, and_mat);
	return (sum(xor_mat)[0] - 2 * sum(and_mat)[0]) / 255;
}

// lower score is better.
// "shift" and "theta" are corrections for edge 2.
// edge2 images are rotated ~180 degrees for comparison.
EdgeMatch EdgeOfPiece::matchEdges(EdgeOfPiece edge1, EdgeOfPiece edge2, bool verbose) {

	int minHeight = min(edge1.edgeImg.rows, edge2.edgeImg.rows);
	int maxHeight = max(edge1.edgeImg.rows, edge2.edgeImg.rows);
	int h_intervals = (maxHeight - minHeight) / pixelShift;
	int minWidth = min(edge1.edgeImg.cols, edge2.edgeImg.cols);
	int windowWidth = minWidth * 8 / 10;
	int maxWidth = max(edge1.edgeImg.cols, edge2.edgeImg.cols);
	int w_intervals = (maxWidth - windowWidth) / pixelShift;

	double minScore;
	int bestTheta;
	Point bestShift;
	Mat best_e1; // for display only
	Mat best_e2; // for display only
	bool firstMatch = true;

	for(int i = 0; i < edge2.rotEdgeImgAngles.size(); i++) {

		double theta = edge2.rotEdgeImgAngles[i];
		Mat rotEdgeImg = edge2.rotEdgeImgs[i];

		for(int h = 0; h < h_intervals + 1; h++) {
			for(int w = 0; w < w_intervals + 1; w++) {
				Range e1ColRange;
				Range e2ColRange;
				Range e1RowRange;
				Range e2RowRange;

				if(edge1.edgeImg.rows <= edge2.edgeImg.rows) {
					e1RowRange = Range(0, minHeight);
					e2RowRange = Range(h * pixelShift, h * pixelShift + minHeight);
				} else {
					e2RowRange = Range(0, minHeight);
					e1RowRange = Range(h * pixelShift, h * pixelShift + minHeight);
				}
				if(edge1.edgeImg.cols <= edge2.edgeImg.cols) {
					e1ColRange = Range(minWidth/10, minWidth/10 + windowWidth);
					e2ColRange = Range(w * pixelShift, w * pixelShift + windowWidth);
				} else {
					e2ColRange = Range(minWidth/10, minWidth/10 + windowWidth);
					e1ColRange = Range(w * pixelShift, w * pixelShift + windowWidth);
				}
				Mat e1 = edge1.edgeImg.rowRange(e1RowRange).colRange(e1ColRange);
				Mat e2 = rotEdgeImg.rowRange(e2RowRange).colRange(e2ColRange);

				double score = edgeComparisonScore(e1, e2);

				if(firstMatch || score < minScore) {
					if(firstMatch) firstMatch = false;
					minScore = score;
					bestTheta = theta;
					// (1) apply edge2 raster shift, but fliped across center row bc edge2 is rotated 180 degrees.
					// (2) apply the shift calculated in this function
					// (3) apply reverse of edge1 raster shift
					bestShift = Point(edge2.rasterShift.x, rotEdgeImg.rows - edge2.rasterShift.y) + Point(e1ColRange.start - e2ColRange.start, e1RowRange.start - e2RowRange.start) - edge1.rasterShift;
					best_e1 = e1;  // for display
					best_e2 = e2;
				}
			}
		}
	}

	if(verbose) {
		cout << "best theta: " << bestTheta << endl;

		Mat bothEdges;
		Mat channels[3] = {best_e1, Mat::zeros(best_e1.size(), CV_8UC1), best_e2};
		merge(channels, 3, bothEdges);

		namedWindow("edgeMatch");
		imshow("edgeMatch", bothEdges);
		waitKey(0);
		destroyWindow("edgeMatch");
	}

	EdgeMatch bestMatch;
	bestMatch.score = minScore;
	bestMatch.theta = bestTheta;
	bestMatch.shift = bestShift * edgeShrinkFactor;
	return bestMatch;
}

// search through remaining pieces and return best match for edge "edgeIndex" of piece "piece"
PieceMatch Puzzle::match(PuzzlePiece *piece, int edgeIndex, bool edgesOnly, bool verbose) {

	PieceMatch bestMatch;
	double bestMatchScore;
	bool firstMatch = true;

	if(piece->edges[edgeIndex].isEdge) {
		cout << "ERROR: calling match() on an edge piece" << endl;
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

	cout << "Piece " << piece->number << " matches edge " << bestMatch.edgeIndex << " of piece " << bestMatch.piece->number << endl;
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
			cursor->correctionShiftLeft = Point(-matchingPieces[0].match.shift.y, matchingPieces[0].match.shift.x);  // // rotate clockwise 90deg
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
