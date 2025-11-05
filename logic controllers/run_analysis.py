#!/usr/bin/env python3
"""
Quick analysis runner for real Stake data
"""

from real_stake_analyzer import RealStakeAnalyzer
import time

def run_complete_analysis():
    print('🎯 GENERATING COMPLETE STAKE ANALYSIS')
    print('====================================')

    analyzer = RealStakeAnalyzer()

    print('Step 1: Generating 1,629 bet history...')
    start_time = time.time()
    history = analyzer.generate_complete_history()
    print(f'✅ History generated in {time.time() - start_time:.1f}s')

    print('\nStep 2: Analyzing sharp patterns...')
    start_time = time.time()
    patterns = analyzer.analyze_sharp_patterns()
    print(f'✅ Pattern analysis completed in {time.time() - start_time:.1f}s')

    print('\nStep 3: Predicting next 20 bets...')
    start_time = time.time()
    predictions = analyzer.predict_next_sequence(20)
    print(f'✅ Predictions completed in {time.time() - start_time:.1f}s')

    print('\n🎯 READY FOR HIGH-STAKES BETTING!')
    print('All analysis complete. Key insights:')
    
    results = [h['result'] for h in history]
    print(f'- Historical range: {min(results):.1f} - {max(results):.1f}')
    print(f'- Average result: {sum(results)/len(results):.1f}')
    print(f'- Sharp movements: {len(patterns["sharp_jumps"]) + len(patterns["sharp_drops"])}')
    print(f'- Next predictions ready for betting!')
    
    # Show some key predictions
    print('\n🔮 KEY UPCOMING PREDICTIONS:')
    for i, pred in enumerate(predictions[:5]):
        nonce = pred['nonce']
        result = pred['predicted_result']
        print(f'  Prediction {i+1}: Nonce {nonce} → {result:.2f}')
    
    # Show betting opportunities  
    print('\n💰 IMMEDIATE BETTING OPPORTUNITIES:')
    for i, pred in enumerate(predictions[:5]):
        result = pred['predicted_result']
        if result <= 20:
            print(f'  🔥 Prediction {i+1}: STRONG UNDER 25 bet ({result:.1f})')
        elif result >= 80:
            print(f'  🔥 Prediction {i+1}: STRONG OVER 75 bet ({result:.1f})')
        elif 45 <= result <= 55:
            print(f'  ⚡ Prediction {i+1}: RANGE 45-55 bet ({result:.1f})')
    
    # Save analysis
    filename = analyzer.save_analysis()
    print(f'\n💾 Complete analysis saved to: {filename}')
    
    return analyzer, history, patterns, predictions

if __name__ == "__main__":
    run_complete_analysis()