#ifndef INCLUDE_HANDLER_H
#define INCLUDE_HANDLER_H

#include "protobuf/Minimap.pb.h"
#include "protobuf/Moving.pb.h"
#include "Message.hpp"
#include "conmmunicator/communicator.hpp"

#define MINIMAP 0
#define MOVING 1

class Package
{
private:
    /* package head and tail which are set manually */
    const int package_head = 0x10011001;
    const int package_tail = 0x10011001;

public:
    /* all data contained in the package */
    int head;
    int commandId;
    int length;
    char *data;
    int tail;
    /* error */
    bool error;

    Package(const char buffer[1024],const int size)
    {
        head = byteToint(buffer[0], buffer[1], buffer[2], buffer[3]);
        commandId = byteToint(buffer[4], buffer[5], buffer[6], buffer[7]);
        length = byteToint(buffer[8], buffer[9], buffer[10], buffer[11]);
        data = new char[length-16];
        strncpy(data, &(buffer[12]), length-16);
        tail = byteToint(buffer[length-4], buffer[length-3], buffer[length-2], buffer[length-1]);
        if(length != size || head != package_head || tail != package_tail)
            error = true;
        else
            error = false;
    }
    ~Package(){}
    int byteToint(char byte1, char byte2, char byte3, char byte4)
    {
        int num = byte4 & 0xFF;
        num |= ((byte3 << 8) & 0xFF00);
        num |= ((byte2 << 16) & 0xFF0000);
        num |= ((byte1 << 24) & 0xFF0000);
        return num;
    }
};

class Handler
{
private:
    /* data */
    armor::CommunicatorUSB communicator;

public:
    bool error;

    Handler(/* args */){
        communicator.open(0x0477, 0x5620);
        communicator.enableReceiveGlobalAngle(true);
        //communicator.startReceiveService();
    };
    ~Handler(){};
    
    int handle_message(const char buffer[1024],const int size)
    {
        Package package(buffer, size);
        if(package.error)
            return -1;
        if(package.commandId == MINIMAP)
            handle_minimap(package.data);
        else if(package.commandId == MOVING)
            handle_moving(package.data);
        return 0;
    }

    void handle_minimap(const char *data)
    {
        Transfer::Minimap minimap;
        minimap.ParseFromString(data);
        MinimapMsg minimapMsg;
        generateMinimapMsg(minimap, minimapMsg);
        communicator.send(minimapMsg, armor::SEND_STATUS_AUTO_AIM, armor::SEND_STATUS_WM_PLACEHOLDER);
    }

    void handle_moving(const char *data)
    {
        Transfer::Moving moving;
        moving.ParseFromString(data);
        MovingMsg movingMsg;
        generateMovingMsg(moving, movingMsg);
        communicator.send(movingMsg, armor::SEND_STATUS_AUTO_AIM, armor::SEND_STATUS_WM_PLACEHOLDER);
    }

    void generateMinimapMsg(const Transfer::Minimap &minimap, MinimapMsg &minimapMsg)
    {
        minimapMsg.cmd = MINIMAP;
        minimapMsg.friendNumber = minimap.friendnumber();
        minimapMsg.fx1 = minimap.fpositions(0).x();
        minimapMsg.fy1 = minimap.fpositions(0).y();
        minimapMsg.fx2 = minimap.fpositions(1).x();
        minimapMsg.fy2 = minimap.fpositions(1).y();
        minimapMsg.fx3 = minimap.fpositions(2).x();
        minimapMsg.fy3 = minimap.fpositions(2).y();
        minimapMsg.fx4 = minimap.fpositions(3).x();
        minimapMsg.fy4 = minimap.fpositions(3).y();
        minimapMsg.fx5 = minimap.fpositions(4).x();
        minimapMsg.fy5 = minimap.fpositions(4).y();
        minimapMsg.fx6 = minimap.fpositions(5).x();
        minimapMsg.fy6 = minimap.fpositions(5).y();
        minimapMsg.fx7 = minimap.fpositions(6).x();
        minimapMsg.fy7 = minimap.fpositions(6).y();
        minimapMsg.fx8 = minimap.fpositions(7).x();
        minimapMsg.fy8 = minimap.fpositions(7).y();
        minimapMsg.enemyNumber = minimap.enemynumber();
        minimapMsg.ex1 = minimap.epositions(0).x();
        minimapMsg.ey1 = minimap.epositions(0).y();
        minimapMsg.ex2 = minimap.epositions(1).x();
        minimapMsg.ey2 = minimap.epositions(1).y();
        minimapMsg.ex3 = minimap.epositions(2).x();
        minimapMsg.ey3 = minimap.epositions(2).y();
        minimapMsg.ex4 = minimap.epositions(3).x();
        minimapMsg.ey4 = minimap.epositions(3).y();
        minimapMsg.ex5 = minimap.epositions(4).x();
        minimapMsg.ey5 = minimap.epositions(4).y();
        minimapMsg.ex6 = minimap.epositions(5).x();
        minimapMsg.ey6 = minimap.epositions(5).y();
        minimapMsg.ex7 = minimap.epositions(6).x();
        minimapMsg.ey7 = minimap.epositions(6).y();
        minimapMsg.ex8 = minimap.epositions(7).x();
        minimapMsg.ey8 = minimap.epositions(7).y();
    }

    void generateMovingMsg(const Transfer::Moving &moving, MovingMsg &movingMsg)
    {
        movingMsg.cmd = MOVING;
        movingMsg.x = moving.x();
        movingMsg.y = moving.y();
    }

};



#endif