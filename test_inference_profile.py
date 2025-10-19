"""
Quick test for AWS Bedrock Inference Profile
Tests if the inference profile works before making changes to config
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_inference_profile(profile_id, region='us-east-1'):
    """
    Test if an inference profile works.
    
    Args:
        profile_id: Inference profile ID to test
        region: AWS region
    """
    print("\n" + "="*60)
    print(f"Testing Inference Profile")
    print("="*60)
    print(f"Profile ID: {profile_id}")
    print(f"Region: {region}")
    print("-"*60)
    
    try:
        # Initialize Bedrock Runtime client
        client = boto3.client('bedrock-runtime', region_name=region)
        
        # Create a minimal test request
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello from Bedrock!' and nothing else."
                }
            ],
            "max_tokens": 20,
            "temperature": 0
        }
        
        print("\n📤 Sending test request...")
        
        # Try to invoke the model using inference profile
        response = client.invoke_model(
            modelId=profile_id,
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract text
        if 'content' in response_body and len(response_body['content']) > 0:
            text = response_body['content'][0].get('text', '')
            
            print("✅ SUCCESS! Inference profile works!")
            print(f"\n📥 Response from Claude:")
            print(f"   '{text}'")
            print("\n" + "="*60)
            print("🎉 You can use this inference profile!")
            print("="*60)
            print(f"\nUpdate config/config.yaml with:")
            print(f"  model_id: \"{profile_id}\"")
            print("="*60)
            return True
        else:
            print("⚠️  Response received but no content found")
            print(f"Response: {response_body}")
            return False
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        
        print(f"\n❌ FAILED: Cannot use inference profile")
        print(f"   Error Code: {error_code}")
        print(f"   Message: {error_msg}")
        print("\n" + "="*60)
        
        if error_code == 'UnrecognizedClientException':
            print("🔍 Issue: Invalid credentials or permissions")
            print("   - Check AWS credentials are valid")
            print("   - Ensure IAM permissions include 'bedrock:InvokeModel'")
            
        elif error_code == 'AccessDeniedException':
            print("🔍 Issue: Model access not granted")
            print("   - Go to AWS Bedrock Console")
            print("   - Enable access to Claude models")
            
        elif error_code == 'ResourceNotFoundException':
            print("🔍 Issue: Inference profile not found")
            print("   - Check the profile ID is correct")
            print("   - Verify it exists in your region")
            
        elif error_code == 'ValidationException':
            print("🔍 Issue: Invalid request or profile ID")
            print("   - Check the profile ID format")
            
        print("="*60)
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_embedding_with_titan(region='us-east-1'):
    """Test Titan embeddings still work."""
    print("\n" + "="*60)
    print("Testing Titan Embeddings")
    print("="*60)
    
    try:
        client = boto3.client('bedrock-runtime', region_name=region)
        
        body = {
            "inputText": "Test embedding",
            "normalize": True
        }
        
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        
        if len(embedding) > 0:
            print(f"✅ Titan embeddings work!")
            print(f"   Dimension: {len(embedding)}")
            return True
        else:
            print("⚠️  No embedding returned")
            return False
            
    except Exception as e:
        print(f"❌ Titan embeddings failed: {e}")
        return False


def main():
    """Run inference profile tests."""
    print("\n" + "🧪"*30)
    print("INFERENCE PROFILE TEST")
    print("🧪"*30)
    
    # Test both profile formats
    profiles_to_test = [
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "arn:aws:bedrock:us-east-1:258574424891:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0"
    ]
    
    print("\n📋 Will test these profiles:")
    for i, profile in enumerate(profiles_to_test, 1):
        print(f"   {i}. {profile}")
    
    # Test each profile
    success = False
    working_profile = None
    
    for profile in profiles_to_test:
        if test_inference_profile(profile):
            success = True
            working_profile = profile
            break  # Stop after first success
        print()  # Spacing between tests
    
    # Also test embeddings
    print()
    test_embedding_with_titan()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if success:
        print(f"✅ SUCCESS! Use this profile:")
        print(f"\n   {working_profile}")
        print(f"\nTo update your config:")
        print(f"   1. Edit config/config.yaml")
        print(f"   2. Change line 6 to:")
        print(f'      model_id: "{working_profile}"')
        print(f"   3. Run: python test_step3.py")
    else:
        print("❌ Neither inference profile worked")
        print("\nYou need to:")
        print("   1. Enable Claude model access in Bedrock Console")
        print("   2. Run: python diagnose_bedrock.py")
        print("   3. Follow the fix instructions")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()