#!/usr/bin/env python3
"""
Test SAM Ultimate AI with multiple epochs
"""

from sam_ultimate_ai_deployment import SAMUltimateAIDeployment

def test_epochs():
    """Test the system with multiple epochs"""
    print("🧪 TESTING SAM ULTIMATE AI WITH MULTIPLE EPOCHS")
    print("=" * 60)
    print("🎯 Running 3 epochs to demonstrate fixed system")
    print("🌐 Web scraping + Ollama + SAM + Knowledge base")
    print("⏰️ Training Interval: 60 seconds")
    print("🎯 Duration: 3 epochs for testing")
    
    try:
        # Create deployment system
        deployment = SAMUltimateAIDeployment()
        
        # Run deployment with 3 epochs in test mode
        deployment.run_deployment(epochs=3, test_mode=True)
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
    finally:
        print(f"\n🎉 SAM Ultimate AI testing completed!")

if __name__ == "__main__":
    test_epochs()
