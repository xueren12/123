#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据库中的数据"""

from utils.db import TimescaleDB

def main():
    db = TimescaleDB()
    db.connect()
    
    with db.conn.cursor() as cur:
        # 检查 orderbook 表
        cur.execute('SELECT COUNT(*) FROM orderbook')
        result = cur.fetchone()
        total_count = result['count'] if result else 0
        print(f'orderbook表总记录数: {total_count}')
        
        if total_count > 0:
            cur.execute('SELECT inst_id, COUNT(*) FROM orderbook GROUP BY inst_id')
            inst_counts = cur.fetchall()
            print('各交易对记录数:', inst_counts)
            
            # 查看最新的几条记录
            cur.execute('SELECT ts, inst_id FROM orderbook ORDER BY ts DESC LIMIT 5')
            recent = cur.fetchall()
            print('最新5条记录的时间戳:', recent)
        
        # 检查 trades 表
        cur.execute('SELECT COUNT(*) FROM trades')
        result = cur.fetchone()
        trades_count = result['count'] if result else 0
        print(f'trades表总记录数: {trades_count}')
    
    db.close()

if __name__ == '__main__':
    main()