//
//created by 刘雍熙 on 2020-02-01
//

#ifndef COMMUNICATE_HPP
#define COMMUNICATE_HPP

#include <cmath>

#include <serial/serial.h>

#include "crc_table.hpp"
#include "usbio.hpp"

#include "../include/Message.hpp"

namespace armor{

    /**
     * 通讯基类
     */
    class Communicator{
        protected:

#pragma region Self_defined data
#pragma pack(1)
        //头帧 时间戳 yaw pitch 标识符 校验 尾帧
        struct __MinimapFrameSt
        {
            uint8_t head = 0xf1;
            int cmd = 0;
            int friendNumber = 0;
            int fx1 = 0;
            int fy1 = 0;
            int fx2 = 0;
            int fy2 = 0;
            int fx3 = 0;
            int fy3 = 0;
            int fx4 = 0;
            int fy4 = 0;
            int fx5 = 0;
            int fy5 = 0;
            int fx6 = 0;
            int fy6 = 0;
            int fx7 = 0;
            int fy7 = 0;
            int fx8 = 0;
            int fy8 = 0;
            int enemyNumber = 0;
            int ex1 = 0;
            int ey1 = 0;
            int ex2 = 0;
            int ey2 = 0;
            int ex3 = 0;
            int ey3 = 0;
            int ex4 = 0;
            int ey4 = 0;
            int ex5 = 0;
            int ey5 = 0;
            int ex6 = 0;
            int ey6 = 0;
            int ex7 = 0;
            int ey7 = 0;
            int ex8 = 0;
            int ey8 = 0;
            uint8_t crc8check = 0;
            uint8_t end = 0xf2;
        } minimap_frame;
#pragma pack()
#pragma pack(1)
        //头帧 时间戳 yaw pitch 标识符 校验 尾帧
        struct __MovingFrameSt
        {
            uint8_t head = 0xf1;
            int cmd = 0;
            int x = 0;
            int y = 0;
            uint8_t crc8check = 0;
            uint8_t end = 0xf2;
        } moving_frame;
#pragma pack()
#pragma endregion

        //crc校验用
        static uint8_t m_calcCRC8(const uint8_t *buff, size_t len) {
            uint8_t ucIndex, ucCRC8 = (uint8_t) CRC8_INIT;
            while (len--) {
                ucIndex = ucCRC8 ^ (*buff++);
                ucCRC8 = CRC8_Table[ucIndex];
            }
            return (ucCRC8);
        }

        const size_t m_minimap_frameSize = sizeof(__MinimapFrameSt);
        const size_t m_moving_frameSize = sizeof(__MovingFrameSt);

        public:
        //构造函数
        explicit Communicator()
        {
            
        }

        ~Communicator() {}

    };

    /**
     * USB 通讯子类
     */

    class CommunicatorUSB : public Communicator {
        private:
        superpower::usbio::spUSB *m_usb;

        public:
        explicit CommunicatorUSB(): m_usb(nullptr){}

        /**
         * 打开设备, 带后台线程
         * @param vid 0x0477 16进制
         * @param pid 0x5620 16进制
         */
        void open(int vid, int pid) {
            m_usb = new superpower::usbio::spUSB(vid, pid);
        }

        void send(float rYaw, float rPitch) override {
            /* 刷新结构体 */
            if (m_frame.timeStamp > 0xfffe) m_frame.timeStamp = 0;
            m_frame.timeStamp++;
            m_frame.yaw = rYaw;
            m_frame.pitch = rPitch;
            m_frame.extra[0] = extra0;
            m_frame.extra[1] = extra1;

            /* 组织字节 */
            uint8_t msg[m_frameSize];
            memcpy(msg, &m_frame, m_frameSize);
            msg[m_frameSize - 2] = m_calcCRC8(msg, m_frameSize - 2);

            /* 发送 */
            int len;
            m_usb->write(msg, m_frameSize, len);
        }

        /**
         * send message for moving
         * @param movingMsg a struct contained data for struct minimap_frame
         */
        void send(MovingMsg movingMsg)
        {
            /* update information */
            memcpy((char *)(&moving_frame)+1, &movingMsg, sizeof(movingMsg));

            /* organize bytes */
            uint8_t msg[m_moving_frameSize];
            memcpy(msg, &moving_frame, m_moving_frameSize);
            msg[m_moving_frameSize - 2] = m_calcCRC8(msg, m_moving_frameSize - 2);

            /* sending */
            int len;
            m_usb->write(msg, m_moving_frameSize, len);
        }

        void send(MinimapMsg minimapMsg)
        {
            /* update information */
            memcpy((char *)(&minimap_frame)+1, &minimapMsg, sizeof(minimapMsg));

            /* organize bytes */
            uint8_t msg[m_moving_frameSize];
            memcpy(msg, &moving_frame, m_moving_frameSize);
            msg[m_moving_frameSize - 2] = m_calcCRC8(msg, m_moving_frameSize - 2);

            /* sending */
            int len;
            m_usb->write(msg, m_moving_frameSize, len);
        }
    };
}

#endif //COMMUNICATE_HPP