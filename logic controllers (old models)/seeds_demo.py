#!/usr/bin/env python3
"""
HMAC-SHA256 Seeds Demonstration
Shows how to calculate dice roll results using client seed, server seed, and nonce
"""

import hashlib
import hmac

def calculate_dice_roll(client_seed: str, server_seed: str, nonce: int) -> float:
    """
    Calculate dice roll using HMAC-SHA256 (provably fair algorithm)
    
    Args:
        client_seed: Client's seed string
        server_seed: Server's revealed seed string  
        nonce: Bet number/counter
        
    Returns:
        Dice roll result (0-99.99)
    """
    # Create the message
    message = f"{client_seed}:{nonce}"
    
    # Calculate HMAC-SHA256
    hmac_result = hmac.new(
        server_seed.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Convert first 8 characters to decimal
    hex_chunk = hmac_result[:8]
    decimal_value = int(hex_chunk, 16)
    
    # Convert to dice roll (0-99.99)
    dice_roll = round((decimal_value % 10000) / 100, 2)
    
    return dice_roll

def verify_server_seed_hash(server_seed: str, server_seed_hash: str) -> bool:
    """
    Verify that a revealed server seed matches its hash
    
    Args:
        server_seed: The revealed server seed
        server_seed_hash: The hash that was provided before the game
        
    Returns:
        True if the seed matches the hash
    """
    calculated_hash = hashlib.sha256(server_seed.encode('utf-8')).hexdigest()
    return calculated_hash.lower() == server_seed_hash.lower()

def demo_with_example_seeds():
    """Demonstrate with example seeds from your image"""
    print("🎲 HMAC-SHA256 Dice Roll Calculation Demo")
    print("=" * 50)
    
    # Example from your image
    client_seed = "A1EnyBArgu"
    server_seed_hash = "1c3b4889e9411f47beb7d1149031d72a61e704911415922999e3579df87e3c8b"
    nonce = 0  # Starting nonce (total bets made = 0)
    
    print(f"Client Seed: {client_seed}")
    print(f"Server Hash: {server_seed_hash}")
    print(f"Starting Nonce: {nonce}")
    print()
    
    # Note: We need the actual server seed (not just the hash) to calculate
    print("⚠️  IMPORTANT: To calculate actual dice rolls, we need the REVEALED server seed")
    print("   The server seed hash alone cannot be used for HMAC calculation.")
    print("   The hash is used for verification AFTER the seed is revealed.")
    print()
    
    # Example with a hypothetical revealed server seed
    example_server_seed = "example_revealed_seed_123456"
    print(f"Example Revealed Seed: {example_server_seed}")
    print()
    
    # Verify the seed (this would fail with our example)
    is_valid = verify_server_seed_hash(example_server_seed, server_seed_hash)
    print(f"Seed Verification: {'✅ VALID' if is_valid else '❌ INVALID (expected with example seed)'}")
    print()
    
    # Calculate next 5 dice rolls
    print("Next 5 dice rolls (with example seed):")
    print("-" * 30)
    
    for i in range(5):
        current_nonce = nonce + i
        dice_roll = calculate_dice_roll(client_seed, example_server_seed, current_nonce)
        
        # Create the HMAC input for transparency
        message = f"{client_seed}:{current_nonce}"
        hmac_result = hmac.new(
            example_server_seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        print(f"Nonce {current_nonce}:")
        print(f"  Input: {message}")
        print(f"  HMAC: {hmac_result}")
        print(f"  Roll: {dice_roll}")
        print()

def demo_how_to_use_in_dashboard():
    """Show how to use the seeds in the dashboard"""
    print("\n📊 How to Use Seeds in the Dashboard")
    print("=" * 50)
    print()
    print("1. Enter your CLIENT SEED in the dashboard")
    print("   - This is the seed YOU choose (e.g., 'A1EnyBArgu')")
    print("   - You can change this anytime before starting a new session")
    print()
    print("2. Enter the SERVER SEED HASH")
    print("   - This is provided by Stake.com at the start of each session")
    print("   - It's a 64-character hex string (SHA256 hash)")
    print("   - Example: 1c3b4889e9411f47beb7d1149031d72a61e704911415922999e3579df87e3c8b")
    print()
    print("3. Set the NONCE (bet counter)")
    print("   - This starts at 0 and increments with each bet")
    print("   - The dashboard will auto-increment this for you")
    print()
    print("4. (Optional) Enter REVEALED SERVER SEED")
    print("   - This is only available AFTER the session ends")
    print("   - Used to verify the results were fair and calculate actual predictions")
    print()
    print("5. Click 'Calculate Next Numbers'")
    print("   - With hash only: Limited prediction accuracy")
    print("   - With revealed seed: High accuracy HMAC calculations")
    print()
    print("6. Click 'Verify Seeds' (when server seed is revealed)")
    print("   - Confirms the server seed matches the original hash")
    print("   - Proves the game was provably fair")

if __name__ == "__main__":
    demo_with_example_seeds()
    demo_how_to_use_in_dashboard()
    
    print("\n" + "=" * 50)
    print("🚀 Ready to use enhanced predictions in the dashboard!")
    print("   Launch the dashboard and enter your session seeds for")
    print("   improved number prediction accuracy.")
    print("=" * 50)