#ifndef GENSFEN_H_INCLUDED
#define GENSFEN_H_INCLUDED

#include <string>

#include "board.h"

struct gensfenData
{
    std::string fen;
    int eval;
    double result;
};

// void gensfenCommand(Board &b, const std::vector<std::string> &words)
// {
//     //generate self-play data.
//     if (words.size() != 11) {return;}
//     if (words[1] != "depth" || words[3] != "positions" ||
//         words[5] != "randomply" || words[7] != "maxply" ||
//         words[9] != "evalbound")
//     {
//         return;
//     }
//     if (!isNumber(words[2]) || !isNumber(words[4]) ||
//         !isNumber(words[6]) || !isNumber(words[8]) ||
//         !isNumber(words[10]))
//     {
//         return;
//     }

//     std::string dateTime = std::format("{:%F_%H-%M-%S_%Z}",
//         std::chrono::zoned_time{
//             std::chrono::current_zone(),
//             std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now())
//         }
//     );

//     int depth = std::stoi(words[2]);
//     int positions = std::stoi(words[4]);
//     int randomply = std::stoi(words[6]);
//     int maxply = std::stoi(words[8]);
//     int evalbound = std::stoi(words[10]);

//     std::vector<gensfenData> output = {};
//     std::vector<gensfenData> outputBuffer = {};

//     //set up random device.
//     std::random_device _rd;
//     std::size_t seed;

//     if (_rd.entropy()) {seed = _rd();}
//     else {seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();}

//     std::mt19937 _mt(seed);

//     //set up search params.
//     timeLeft = INT_MAX;
//     isSearchAborted = false;

//     int numGames = 1;

//     std::cout << "Generating " << positions << " positions..." << std::endl;

//     while ((int)output.size() < positions)
//     {
//         //start a new game.
//         bool gameover = false;
//         double result = 0.5;
//         prepareForNewGame(b);
//         b.setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//         outputBuffer.clear();

//         //track zHash for draw by repetition.
//         std::set<U64> zHash;

//         //play random moves up to randomply.
//         for (int i=0;i<randomply;i++)
//         {
//             b.moveBuffer.clear();
//             b.generatePseudoMoves();

//             //if no moves left we abort the game.
//             if (b.moveBuffer.size() == 0) {gameover = true; break;}

//             std::uniform_int_distribution<int> _dist(0, (int)b.moveBuffer.size() - 1);
//             U32 move = b.moveBuffer[_dist(_mt)];
//             b.makeMove(move);

//             //store zHash.
//             zHash.insert(b.zHashPieces ^ b.zHashState);
//         }

//         if (gameover) {continue;}

//         //fixed-depth search.
//         while (true)
//         {
//             int score = alphaBetaRoot(b, depth, true);

//             //check that score within bounds.
//             if (b.side) {score *= -1;}
//             if (abs(score) > evalbound || isGameOver)
//             {
//                 result = score == 0 ? 0.5 :
//                          score > evalbound ? 1. : 0.;
//                 break;
//             }

//             //exit if maxply exceeded.
//             if ((int)b.moveHistory.size() > maxply)
//             {
//                 result = score > 400 ? 1. :
//                          score < -400 ? 0. : 0.5;
//                 break;
//             }

//             //update output buffer.
//             U32 pieceType = (storedBestMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
//             U32 capturedPieceType = (storedBestMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
//             U32 finishPieceType = (storedBestMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
//             bool isQuiet = !util::isInCheck(b.side, b.pieces, b.occupied) && capturedPieceType == 15 && pieceType == finishPieceType;
//             if (isQuiet) {outputBuffer.push_back(gensfenData(positionToFen(b.pieces, b.current, b.side), score, 0.5));}

//             b.makeMove(storedBestMove);

//             //check if position is repeated and remove last output if so.
//             if (zHash.find(b.zHashPieces ^ b.zHashState) != zHash.end())
//             {
//                 if (isQuiet) {outputBuffer.pop_back();}
//                 result = 0.5;
//                 break;
//             }
//             zHash.insert(b.zHashPieces ^ b.zHashState);
//         }

//         //update contents of buffer with game result.
//         for (auto &x: outputBuffer) {x.result = result;}

//         //add contents of buffer to output.
//         for (const auto &x: outputBuffer) {output.push_back(x);}

//         std::cout << "Finished game " << numGames << "; " << output.size() << " positions overall" << std::endl;
//         numGames++;
//     }

//     //trim size of output.
//     while ((int)output.size() > positions) {output.pop_back();}

//     //write output to file.
//     std::string fileName = "gensfen_"+ ENGINE_NAME_NO_SPACE +
//                            "_n" + std::to_string(positions) +
//                            "_d" + std::to_string(depth) +
//                            "_r" + std::to_string(randomply) +
//                            "_m" + std::to_string(maxply) +
//                            "_b" + std::to_string(evalbound) +
//                            "_" + dateTime +
//                            ".txt";

//     std::ofstream file;
//     file.open(fileName);

//     for (const auto &x: output)
//     {
//         file << x.fen << "; " << x.eval << "; " << x.result << std::endl;
//     }

//     file.close();

//     std::cout << "Finished generating positions." << std::endl;

//     return;
// }

#endif // GENSFEN_H_INCLUDED
