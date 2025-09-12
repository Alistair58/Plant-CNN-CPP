#if PROFILING

#ifndef TIMER_HPP
#define TIMER_HPP
    

#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>
#include "utils.hpp"

using Clock = std::chrono::steady_clock;


class Timer{
    private:
        std::string name;
        Clock::time_point startTime;
        std::vector<std::unique_ptr<Timer>> childTimers;
        int64_t cumulativeTimeTakenUs = -1;
        uint64_t numUsages = 0;
        bool inProgress = false;
        std::string note = "";
        void reuse(){
            if(this->inProgress){
                throw std::runtime_error("Timer cannot be reused if currently timing");
            }
            this->startTime = Clock::now();
            this->numUsages++;
            this->inProgress = true;
        }
    public:
        Timer(){};

        Timer(std::string timerName,std::string noteInput=""){
            this->name = timerName;
            this->numUsages = 1;
            this->inProgress = true;
            this->startTime = Clock::now();
            this->note = noteInput;
        }
        void stop(std::string noteInput=""){
            if(this->cumulativeTimeTakenUs==-1) this->cumulativeTimeTakenUs = 0;
            this->cumulativeTimeTakenUs += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-startTime).count();
            this->inProgress = false;
            if(noteInput.size()>0){
                this->note = noteInput;
            }
        }

        Timer *addChildTimer(std::string childTimerName,std::string noteInput=""){
            for(int i=0;i<childTimers.size();i++){
                if(childTimers[i]->name==childTimerName && childTimers[i]->note==noteInput){
                    childTimers[i]->reuse();
                    return childTimers[i].get();
                }
            }
            childTimers.emplace_back(std::make_unique<Timer>(childTimerName,noteInput));
            return childTimers.back().get();
        }

        void output(){
            //It needs a reference to recursively pass down
            std::string outputString = "";
            output(outputString,"");
        }

        void output(std::string& result,std::string indentation = ""){
            if(this->inProgress){
                throw std::runtime_error("Timer cannot be ouput whilst timing");
            }
            float meanTimeTakenMs = (float)cumulativeTimeTakenUs/numUsages/1000;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << meanTimeTakenMs;  //3 decimal places
            std::string timerResult = name + ": " + oss.str() + "ms"+((this->note.length()>0)?"  - "+this->note:"");
            std::string indentColour = ANSI_COLOURS[indentation.length()%NUM_ANSI_COLOURS];
            result += "\n"+indentColour+indentation+timerResult+ANSI_RESET;
            for(std::unique_ptr<Timer>& child:childTimers){
                child->output(result,indentation+"-");
            }
            if(indentation.length()==0){
                std::cout << result << std::endl;
            }
        }

        void setNote(std::string noteInput){
            this->note = noteInput;
        }
};
#endif

#endif